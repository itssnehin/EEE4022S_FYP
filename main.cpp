#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <armadillo>
#include <string>
#include <vector>
#include <bits/stdc++.h>
#include "Timer.h"
// g++ batches.cpp main.cpp -o example -O2 -larmadillo
using namespace std;
using namespace arma;


struct Process {

	int Fs, c;
	int freq, t;
	int nSamples, interval;


	int nBatches;
	int samples;

	int batchStrideNSamples;
	int batchNSamples;
	int nRangeBins;
	int nDopplerBins;

	int nSampBatches;
	int ARDMaxRange_m, ARDMaxDoppler_Hz;
	int TxToRefRxDistance_m;

};

struct Cancellation 
{
	cx_mat R;
	cx_mat S;
};

cx_mat circshift(cx_mat in, int xshift, int yshift)
{
    int ydim = in.n_cols;
    int xdim = in.n_rows;
    cx_mat out(size(in));

    // circshift each row of mat in
    for (int i = 0; i < xdim; ++i)
    {
    	int ii = (i + xshift) % xdim;
    	for (int j = 0; j < ydim; ++j)
    	{
    		int jj = (j + yshift) % ydim;
    		out(ii,jj) = in(i,j);
    	}
    }

    return out;
}

/*
	Function to perform matlab's equivalent of fftshift(X)
	Where X is some vector
	Swap left and right halves of X
		centre point belongs to the left
*/

cx_rowvec fftshift(cx_rowvec X)
{
	// check even or odd length
	int left_size;
	int length = X.n_elem;

	if (length % 2 == 0)
	{
		left_size = length/2 - 1;
	} else {
		left_size = length/2;
	}


	cx_rowvec left = X.subvec(0, left_size);
	cx_rowvec right = X.subvec(left_size+1, length-1);

	cx_rowvec result = join_rows(right, left);

	return result;

}


/*
	Implement ECA_CD cancellation
*/
Cancellation ECA_CD(cx_vec r_ch, cx_vec s_ch, Process proc)
{

	// form matrix r, s
	// don't transpose since armadillo iterates by col 
	cx_mat r = reshape(r_ch, proc.interval / proc.nSamples, proc.nSamples);
	cx_mat s = reshape(s_ch, proc.interval / proc.nSamples, proc.nSamples);	

	// Transform pulses into freq domain
	// fft of each col

	Cancellation result;
	result.S = fft(s).t();
	result.R = fft(r).t();

	// for no cancellations
	//cout << "No cancellation" << endl;
	//return result;

	double fd = 1*(1/proc.t);


	// Make a phase shifting matrix
	//constants
	cx_double j(0.0,1.0);
	const double PI = 3.141592653589793238463;

	cx_vec phase_shift_vec(proc.nSamples);

	#pragma omp parallel for
	for (int i = 0; i < proc.nSamples; ++i)
	{
		phase_shift_vec(i) = exp(j*PI*((double)2*fd*i*proc.t/proc.nSamples));
	}

	cx_mat L = diagmat(phase_shift_vec);

	#pragma omp parallel for
	for (int i = 0; i < proc.interval/proc.nSamples; ++i)
	{
		cx_vec q = result.R.col(i);

		cx_mat Q = join_horiz(L.t()*q, q, L*q);

		cx_vec y = result.S.col(i);

		//this line
		cx_vec z = (eye(proc.nSamples, proc.nSamples) - Q*inv(Q.t()*Q)*Q.t())*y;
		result.S.col(i) = z;
	}


	return result;
}

cx_mat CAF_batches_alt(cx_mat R, cx_mat S, Process proc)
{
	cx_mat H = S % conj(R); //cross correlation

	H = H.t(); // column = old rows
	cx_mat h_mat = ifft(H); //ifft of each column
	h_mat = h_mat.t(); // transpose again to get rows
	
	// fftshift(h_mat, 2) swap halves for each row
	#pragma omp parallel for
	for (int i = 0; i < h_mat.n_rows; ++i)
	{
		h_mat.row(i) = fftshift(h_mat.row(i));
	}

	int p = floor(h_mat.n_cols/2);
	cx_mat caf = conj(h_mat.cols(p, h_mat.n_cols-1)); //select right half of matrix

	cx_mat CAF = fft(caf);

	//fftshift(conj,1) swap each col
	#pragma omp parallel for
	for (int i = 0; i < CAF.n_cols; ++i)
	{
		cx_rowvec r = CAF.col(i).t();
		CAF.col(i) = fftshift(r).t();
	}

	return CAF;

}



mat CFAR(cx_mat CAF, vec kernel, double pfa)
{
	int row_shift = floor(kernel.n_rows/2);
	double N = sum(kernel);

	//cout << "N: " << N << endl;

	double a = N*(pow(pfa,(-1.0/N)) - 1);

	mat CAF_squared = abs(pow(CAF,2));
	//CAF_squared.save("CAF_squared.csv", csv_ascii);
	mat k = zeros(size(CAF));


	k(span(0,kernel.n_rows-1), span(0, kernel.n_cols-1)) = kernel;
	k = k/N;

	cx_mat K = fft2(k);
	cx_mat P = conj(K) % fft2(CAF_squared);
	cx_mat p_unshift = ifft2(P);
	
	cx_mat p = circshift(p_unshift, row_shift, 0);
	//cout << p(span(1, 1), span(1,10)) << endl;
	mat inds(size(CAF_squared));

	mat PA = real(p*a);


	// #pragma omp parallel for
 	for (int i = 0; i < inds.n_rows; ++i)
	{
		for (int j = 0; j < inds.n_cols; ++j)
		{
			if (CAF_squared(i,j) > PA(i,j))
			{
				inds(i,j) = 1;
			} else{
				inds(i,j) = 0;
			}
		}
	}



	return inds;

}

int main(int argc, char const *argv[])
{
	// Load the Receiver data
	int dur = 1; // time duration for each recording in seconds
	int elems = dur * 240e3; //num of elements per iteration

	int16_t r_val, s_val;
	ifstream r_file("Channel_0.bin", ios::in | ios::binary);
	ifstream s_file("Channel_1.bin", ios::in | ios::binary);

	if (!r_file or !s_file)
	{
		cout << "Error reading data" << endl;
		return 0;
	}

	cout << "\t ECA \t ARD \t CFAR " << endl;
	
	std::vector<cx_double> r_ch_vec;
	std::vector<cx_double> s_ch_vec;

	cx_double j(0.0, 1.0);

	mat output;
	mat ARD_total;
	for (int run = 0; run < 288; ++run) //288 for full
	{

		cout << run << "\t";
		tic();
		
		// read both files at the same time
		// took longer when threaded
		//#pragma omp parallel for
		for (int file = 0; file < 2; ++file)
		{
			
			if (file == 0)
			{
				int counter = 0;
				for (int i = 0; i < 2*elems; ++i)
				{

					r_file.read(reinterpret_cast<char*>(&r_val), 2);
					if (i%2 == 0) // even sample is real 
					{
						r_ch_vec.push_back((double)r_val);
					} else { // odd sample is imag
						r_ch_vec[counter++] +=  j*(double)r_val;
					}
				}
			} else {

				int counter = 0;
				for (int i = 0; i < 2*elems; ++i)
				{

					
					s_file.read(reinterpret_cast<char*>(&s_val), 2);
					if (i%2 == 0) // even sample is real 
					{
						s_ch_vec.push_back((double)s_val);
					} else { // odd sample is imag
						s_ch_vec[counter++] +=  j*(double)s_val;
					}
				}

			}

		}

		cout << "Done reading data" << endl;

		cx_vec r_ch(r_ch_vec);
		r_ch_vec.clear();
		// cout << r_ch.size() << endl;

		cx_vec s_ch(s_ch_vec);
		s_ch_vec.clear();
		// cout << s_ch.size() << endl;

		Process proc;

		proc.freq = 99.3e6;
		proc.t = dur;
		proc.nSamples = 500;

		proc.Fs = 240e3; // Hz
		proc.interval = proc.t * proc.Fs;

		proc.c = 3e8; // m/s
		proc.nBatches = 100; //vary this
		proc.samples = r_ch.n_elem;

		proc.ARDMaxRange_m = 150000;
		proc.ARDMaxDoppler_Hz = 200; // around here
		proc.TxToRefRxDistance_m = 0;
		proc.nSampBatches = 10;

		tic();
		
		Cancellation c = ECA_CD(r_ch, s_ch, proc);
		
		double time_taken = toc();

		cout << fixed << time_taken << setprecision(5) << "s \t" ;

		tic();
		cx_mat CAF = reverse(CAF_batches_alt(c.R, c.S, proc));

		time_taken = toc();
		mat ARD_out = abs(CAF);

		cout << fixed << time_taken << setprecision(5) << "s \t" ;

		tic();

		double max = ARD_out.max(); // expensive operation

		ARD_out = ARD_out / max;

		ARD_out.save("ARD_out.csv", csv_ascii);
		// implement CFAR algorithm
		vec kernel = {1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1};
		double pfa = 1e-5;

		mat detections = CFAR(CAF, kernel, pfa);

		
		// combine CFAR results
		if (run == 0)
		{
			output = detections;
			//ARD_total = ARD_out;
		} else {
			output = output + detections;
			//ARD_total = ARD_total + ARD_out;
		}

		//ARD_total.save("ARD_total.csv", csv_ascii);
		//detections.save("detections.csv", csv_ascii);

		time_taken = toc();

		cout << fixed << time_taken << setprecision(5) << "s \t" << endl;

		// cout << "Time taken is: " << fixed << time_taken << setprecision(5);
		// cout << " sec " << endl;
	}


	
	mat::iterator it     = output.begin();
	mat::iterator it_end = output.end();

	// parallel this?
	tic();

	/*
	for(; it != it_end; ++it)
	{
  		if ((*it) > 0)
  		{
  			(*it) = 1;
  		} else {
  			(*it) = 0;
  		}
  	}
  	*/
  	
  	double time_taken = toc();

	cout << "Time taken is: " << fixed << time_taken << setprecision(5);
	cout << " sec " << endl;

	output.save("output.csv", csv_ascii);
	return 0;
}