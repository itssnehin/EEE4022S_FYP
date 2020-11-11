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
#include <sys/stat.h>
#include <stdint.h>
#include <endian.h>
// g++ batches.cpp main.cpp -o example -O2 -larmadillo
using namespace std;
using namespace arma;



int main(int argc, char const *argv[])
{
	int dur = 1; // time duration for each recording in seconds
	int elems = dur * 200000; //num of elements

	int16_t r_val, s_val;
	ifstream r_file("Channel_0.bin", ios::in | ios::binary);
	ifstream s_file("Channel_1.bin", ios::in | ios::binary);

	if (!r_file or !s_file)
	{
		cout << "Error reading data" << endl;
		return 0;
	}

	
	std::vector<cx_double> r_ch;
	std::vector<cx_double> s_ch;

	cx_double j(0.0, 1.0);

	// read both files at the same time

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
					r_ch.push_back((double)r_val);
				} else { // odd sample is imag
					r_ch[counter++] +=  j*(double)r_val;
				}
			}
		} else {

			int counter = 0;
			for (int i = 0; i < 2*elems; ++i)
			{

				
				s_file.read(reinterpret_cast<char*>(&s_val), 2);
				if (i%2 == 0) // even sample is real 
				{
					s_ch.push_back((double)s_val);
				} else { // odd sample is imag
					s_ch[counter++] +=  j*(double)s_val;
				}
			}

		}

	}

	cx_vec r_ch_vec(r_ch);
	r_ch.clear();
	cout << r_ch_vec.size() << endl;

	cx_vec s_ch_vec(s_ch);
	s_ch.clear();
	cout << s_ch_vec.size() << endl;

    //arma::Mat<short> mymat(buffer, 10, 3, false);
    //std::cout << mymat << std::endl;
    return 0;

}