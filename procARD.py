import os
import pathlib
import socket as soc
import sys
import numpy as np
import matplotlib.pyplot as plt
import struct
import time
import scipy.signal
import scipy.fftpack
import scipy.io

########################################################

def toInt64(byte):
    # Convert byte data into int64
    int64 = int.from_bytes(byte, byteorder='little')
    return int64

# Clearly toUint32 and toUint64 below are the same. The reason I use both is because the data types
# are technically 32 and 64 and therefore these functions simply highlight that in the case that
# someone wants to improve the code later

def toUint32(byte):
    # Convert byte data into unsigned int32
    uint32 = int.from_bytes(byte, byteorder='little', signed=False)
    return uint32

def toUint64(byte):
    # Convert byte data into unsigned int64
    uint64 = int.from_bytes(byte, byteorder='little', signed=False)
    return uint64

def toFloat32(byte):
    # Convert byte data into float32
    float32 = struct.unpack('f', byte)
    return float32[0]

def plotARD(ard, fig):
    # Plot the ARD received from file or stream
    data = 10*np.log10(ard.dataMatrix)
    
    plt.figure(fig)
    plt.imshow(data, aspect='auto', animated=True)
    plt.grid(alpha=.5)
    plt.xlim(0, ard.XDimension-1)
    plt.show(block=False)
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    plt.title(ard.fileName)
    plt.pause(0.05)

def plotCFAR(ard, detections, fig):
    # Plot CFAR from file or stream
    plt.figure(fig)
    plt.imshow(detections, aspect='auto', animated=True)
    plt.grid(alpha=.5)
    plt.xlim(0, ard.XDimension-1)
    plt.show(block=False)
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    plt.title(ard.fileName)
    plt.pause(0.05)

def cfar(ard, guard, ref, pfa):
    # Calculate CFAR
    # Create the kernel
    kernel = np.ones(((ref+guard)*2+1,1))
    # Replace cells with guard cells
    kernel[ref:ref+guard*2+1] = 0

    # Determine the threshold
    N = np.sum(kernel) # Number of training cells
    alpha = N*(pfa**(-1/N)-1) # Threshold gain
    
    kernel = kernel/np.sum(kernel)
    data = 10*np.log10(ard.dataMatrix)

    # Determine the cell under test matrix
    CUT = scipy.signal.convolve2d(data, kernel, "same")

    # Calculate the threshold
    threshold = CUT + 10*np.log10(alpha)
    
    # Threshold exceedance
    detections = data > threshold

    # Determine the local maximum (centroid)
    detectionMatrix = detections * data

    detectCentroid = []
    detectSNR = []
    Xindex = []
    Yindex = []

    centroidMap = np.zeros((ard.YDimension, ard.XDimension))
    centroidMap_snr = np.zeros((ard.YDimension, ard.XDimension))

    start = time.time()
    # Loop through each doppler line
    for i in range(ard.YDimension):
        # For each range bin in a doppler line
        for j in range(ard.XDimension):
            # If there are more than one positive detection in a row (next to each other)
            if detectionMatrix[i][j] < 0:
                # Create centroid array
                detectCentroid.append(detectionMatrix[i][j])
                detectSNR.append(threshold[i][j])
                Xindex.append(j)

            elif len(detectCentroid) == 0:
                continue

            else:
                # Find the maximum in the centroid array
                centroid = np.max(detectCentroid)
                for index in range(len(detectCentroid)):
                    # Extract the index and the SNR for the centroid
                    if detectCentroid[index] == centroid:
                        centroidMap[i][Xindex[index]] = 1
                        centroidMap_snr[i][Xindex[index]] = centroid - detectSNR[index]

                detectCentroid = []
                detectSNR = []
                Xindex = []
                Yindex = []
                
    # print("total time taken to CFAR: ", time.time() - start)
    
    del data, threshold, CUT, detectionMatrix
    return centroidMap, centroidMap_snr, detections

def cfarDaniel(ard, guard, ref, pfa):
    # Calculate CFAR
    # Create the kernel
    kernel = np.ones(((ref+guard)*2+1,1))
    # Replace cells with guard cells
    kernel[ref:ref+guard*2+1] = 0
    N = np.sum(kernel) # Number of training cells
    
    kernel = kernel/np.sum(kernel)
    
    data = 10*np.log10(ard.dataMatrix)
    # Determine the threshold
    alpha = N*(pfa**(-1/N)-1) # Threshold gain
    
    CUT = scipy.signal.convolve2d(data, kernel, "same")

    threshold = CUT + 10*np.log10(alpha)
    print(threshold)
    
    detections = data > threshold
    return detections

def readARD(strFilename):
    # Read ARD from .ARD file output from CudaProcServer
    # This can be optimised a lot
    file = open(strFilename, 'rb')
    print('Reading data..')
    class ARD:
        # First read the header
        fileName = strFilename
        fileType = file.read(4).decode('ascii')
        ARDType = toInt64(file.read(1))
        TimeStamp_us = toInt64(file.read(8))
        Fc_Hz = toUint32(file.read(4))
        Fs_Hz = toUint32(file.read(4))
        BW_Hz = toUint32(file.read(4))
        MinMapAmplitude = toFloat32(file.read(4))
        MaxMapAmplitude = toFloat32(file.read(4))
        RangeResolution_m = toFloat32(file.read(4))
        DopplerResolution_Hz = toFloat32(file.read(4))
        XDimension = toUint32(file.read(4))
        YDimension = toUint32(file.read(4))
        TxToRefRxDistance_m = toUint32(file.read(4))
        CommentOffset_B = toUint64(file.read(8))
        CommentLength = toUint32(file.read(4))
        FileSize_B = toUint64(file.read(8))
        Comment = 'ARD read from file'
        
        try:
            # Now we can read the actual data
            temp = np.fromfile(file, dtype='<f4', count=XDimension*YDimension)
            dataMatrix = np.reshape(temp, [XDimension, YDimension], order='F').transpose()
            dataMatrix = np.flipud(dataMatrix)
            del temp
        except:
            dataMatrix = np.zeros((YDimension, XDimension))

    file.close()
    print('Data read complete\n')
    return ARD

def findARD(path, i):
    # Loop until a file is found
    print('Looking for file..')
    while True:
        try:
            latestARD = (sorted(pathlib.Path(path).glob('*.ard'))[i])
            print('File found!')
            break
        except:
            time.sleep(0.5)
    return latestARD

def streamLocalARD(IP, port):
    # Stream the data from the LOCAL TCP port i.e. this runs on the machine producing the stream
    print('Collecting next batch')
    # Bind the socket to the port
    server_address = (IP, port)

    # Create a TCP/IP socket
    s = soc.socket(soc.AF_INET, soc.SOCK_STREAM)
    print('Starting up on %s port %s\n' % server_address)
    s.connect(server_address)
    print('Waiting for ARD..')
    
    # Header size is 108
    data = s.recv(108)

    class ARD:
        # We first need to read the header. I found out later that I can use:
        # np.frombuffer(temp, dtype='float32', count=-1) instead of the functions I made
        fileName = data[4:34].decode('ascii')
        fileType = data[35:38].decode('ascii')
        ARDType = toInt64(data[39:40])
        TimeStamp_us = toInt64(data[40:48])
        Fc_Hz = toUint32(data[48:52])
        Fs_Hz = toUint32(data[52:56])
        BW_Hz = toUint32(data[56:60])
        MinMapAmplitude = toFloat32(data[60:64])
        MaxMapAmplitude = toFloat32(data[64:68])
        RangeResolution_m = toFloat32(data[68:72])
        DopplerResolution_Hz = toFloat32(data[72:76])
        XDimension = toUint32(data[76:80])
        YDimension = toUint32(data[80:84])
        TxToRefRxDistance_m = toUint32(data[84:86])
        CommentOffset_B = toUint64(data[88:96])
        CommentLength = toUint32(data[96:100])
        FileSize_B = toUint64(data[100:108])

        # Create dummy matrix
        dataMatrix = np.zeros((XDimension, YDimension))
        # Data is streamed by Doppler bins, we need to take each range bin and pull in all the Doppler values
        for i in range(0, XDimension):
            # Read buffer length of Doppler bins (Bytes)
            temp = s.recv(YDimension * 4)
            #If buffer is not full yet, read the difference and continue
            if len(temp) < YDimension * 4:
                temp = temp + s.recv((YDimension * 4) - len(temp))
                # Convert buffer into float data
                dataMatrix[i:] = np.frombuffer(temp, dtype='float32', count=-1)
            else:
                dataMatrix[i:] = np.frombuffer(temp, dtype='float32', count=-1)

        # Extract the comment
        Comment = s.recv(CommentLength).decode('ascii')

        del temp

        dataMatrix = np.transpose(dataMatrix)

        print('Complete\n')

        s.close()
    del data
    return ARD

def streamRemoteARD(IP, port):
    # Stream the data from the REMOTE TCP port i.e. this runs on an offsite machine
    # Note: I dont know why there needs to be a difference but this is the only way I could get things to work
    print('Collecting next batch')
    # Bind the socket to the port
    server_address = (IP, port)

    # Create a TCP/IP socket
    s = soc.socket(soc.AF_INET, soc.SOCK_STREAM)
    print('Starting up on %s port %s\n' % server_address)
    s.connect(server_address)
    socketOpen = True

    # For reasons not known to me, we need to open the socket, attempt to read the data and then
    # read the data again before we get anything useful, hence the ardReceived flag only iterates once
    ardReceived = False
    # While socket is open, run the following
    print('Waiting for ARD..')
    while socketOpen:
        # Header size is 104
        data = s.recv(104)

        if ardReceived:
            class ARD:
                # We first need to read the header. I found out later that I can use:
                # np.frombuffer(temp, dtype='float32', count=-1) instead of the functions I made
                fileName = data[0:30].decode('ascii')
                fileType = data[31:34].decode('ascii')
                ARDType = toInt64(data[35:36])
                TimeStamp_us = toInt64(data[36:43])
                Fc_Hz = toUint32(data[44:48])
                Fs_Hz = toUint32(data[48:52])
                BW_Hz = toUint32(data[52:56])
                MinMapAmplitude = toFloat32(data[56:60])
                MaxMapAmplitude = toFloat32(data[60:64])
                RangeResolution_m = toFloat32(data[64:68])
                DopplerResolution_Hz = toFloat32(data[68:72])
                XDimension = toUint32(data[72:76])
                YDimension = toUint32(data[76:80])
                TxToRefRxDistance_m = toUint32(data[80:84])
                CommentOffset_B = toUint64(data[84:92])
                CommentLength = toUint32(data[92:96])
                FileSize_B = toUint64(data[96:104])

                # Create dummy matrix
                dataMatrix = np.zeros((XDimension, YDimension))
                # Data is streamed by Doppler bins, we need to take each range bin and pull in all the Doppler values
                for i in range(0, XDimension):
                    # Read buffer length of Doppler bins (Bytes)
                    temp = s.recv(YDimension * 4)
                    #If buffer is not full yet, read the difference and continue
                    if len(temp) < YDimension * 4:
                        temp = temp + s.recv((YDimension * 4) - len(temp))
                        # Convert buffer into float data
                        dataMatrix[i:] = np.frombuffer(temp, dtype='float32', count=-1)
                    else:
                        dataMatrix[i:] = np.frombuffer(temp, dtype='float32', count=-1)

                # Extract the comment
                Comment = s.recv(CommentLength).decode('ascii')

                del temp
                dataMatrix = np.transpose(dataMatrix)

            print('Complete\n')

            socketOpen = False
            s.close()

        ardReceived = True
    del ardReceived, socketOpen, data
    return ARD

def writeSocket(ard, IP, port):
    # Send processed CFAR data to the remote client. This greatly reduces the data rate
    # To be used in conjunction with streamCFAR.py
    print('Sending data')
    # Create a TCP/IP socket
    s = soc.socket(soc.AF_INET, soc.SOCK_STREAM)

    # Bind the socket to the port
    server_address = (IP, port)
    print('Starting up on %s port %s' % server_address)
    s.setsockopt(soc.SOL_SOCKET, soc.SO_REUSEADDR, 1)
    s.bind(server_address)

    # Listen for incoming connections
    s.listen(1)
    print('Waiting for connection')
    connection, client_address = s.accept()
    print('Connection from: %s port %s' % client_address)
##    header = ard.TimeStamp_us.to_bytes(8, byteorder = 'little') + \
##             ard.Fc_Hz.to_bytes(4, byteorder = 'little') + \
##             ard.Fs_Hz.to_bytes(4, byteorder = 'little') + \
##             ard.BW_Hz.to_bytes(4, byteorder = 'little') + \
##             bytearray(struct.pack('<f', ard.RangeResolution_m)) + \
##             bytearray(struct.pack('<f', ard.DopplerResolution_Hz)) + \
##             ard.XDimension.to_bytes(4, byteorder = 'little') + \
##             ard.YDimension.to_bytes(4, byteorder = 'little') + \
##             ard.n.to_bytes(4, byteorder = 'little') + \
##             ard.m.to_bytes(4, byteorder = 'little')
##    connection.sendall(header)
    connection.sendall(ard.TimeStamp_us.to_bytes(8, byteorder = 'little'))
    connection.sendall(ard.Fc_Hz.to_bytes(4, byteorder = 'little'))
    connection.sendall(ard.Fs_Hz.to_bytes(4, byteorder = 'little'))
    connection.sendall(ard.BW_Hz.to_bytes(4, byteorder = 'little'))
    connection.sendall(bytearray(struct.pack('<f', ard.RangeResolution_m)))
    connection.sendall(bytearray(struct.pack('<f', ard.DopplerResolution_Hz)))
    connection.sendall(ard.XDimension.to_bytes(4, byteorder = 'little'))
    connection.sendall(ard.YDimension.to_bytes(4, byteorder = 'little'))
    connection.sendall(ard.n.to_bytes(4, byteorder = 'little'))
    connection.sendall(ard.m.to_bytes(4, byteorder = 'little'))
    
    connection.sendall(ard.detections.flatten('F').tobytes('F'))
    print('Data sent\n')
    # Clean up the connection
    connection.close()

def extractDetections(ard, detections):
    # This extracts the detections from the CFAR that are then streamed to the remote device
    ard.n = 0
    ard.m = 3 # number of columns used
    ard.detections = np.zeros((ard.YDimension * ard.XDimension, ard.m))

    for i in range(ard.XDimension):
        for j in range(ard.YDimension):
            if detections[j][i] or detections[j][i] > 0:
                ard.detections[ard.n][0] = i * ard.RangeResolution_m
                ard.detections[ard.n][1] = j * ard.DopplerResolution_Hz
                ard.detections[ard.n][2] = detections[j][i]
                ard.n = ard.n + 1
    ard.detections = ard.detections[0:ard.n]
    return ard

def buildCFAR(ard, detTot):
    # This is used to build the CFAR back up from the extracted detections
    ard.dataMatrix = np.zeros((ard.YDimension, ard.XDimension))
    ard.dataMatrix_snr = np.zeros((ard.YDimension, ard.XDimension))

    # Reconstruct the original CFAR
    for i in range(ard.n):
        rangeBin = int(ard.detections[i][0]/ard.RangeResolution_m)
        dopplerBin = int(ard.detections[i][1]/ard.DopplerResolution_Hz)
        # Create a normal version
        ard.dataMatrix[dopplerBin][rangeBin] = 1
        # Create a version with the SNR built in
        ard.dataMatrix_snr[dopplerBin][rangeBin] = ard.detections[i][2]
        # Create a combined detection plot
    try:
        detTot = detTot + ard.dataMatrix
    except:
        detTot = ard.dataMatrix
        
    return ard, detTot

def saveCSV(name, ard):
    # Save detections with header as CSV file
    print('Saving CSV')
    name = name + '.csv'

    file = open(name, 'a+')
    file.write('Date' + ',' + \
           'Time' + ',' + \
           'Range [m]' + ',' + \
           'Doppler [Hz]' + ',' + \
           'SNR [dB]' + ',' + \
           'Range Res [m]' + ',' + \
           'Doppler Res [Hz]' + ',' + \
           'Rtot [Bins]' + ',' + \
           'Dtot [Bins]' + '\n')
    for i in range(ard.n):
        file.write(str(ard.dateTime[0:5]) + ',' + \
                   str(ard.dateTime[6:11]) + ',' + \
                   str(ard.detections[i][0]) + ',' + \
                   str(ard.detections[i][1]) + ',' + \
                   str(ard.detections[i][2]) + ',' + \
                   str(ard.RangeResolution_m) + ',' + \
                   str(ard.DopplerResolution_Hz) + ',' + \
                   str(ard.XDimension) + ',' + \
                   str(ard.YDimension) + '\n')
    file.close()
                    
########################################################

# Set streaming parameters
IP = '192.168.10.13'
port_r = 5003
port_s = 1000

# Set path parameters
path = './ARD/CleanSingleTarget/ARDs'

name = 'detections'

# CFAR parameters
guard = 4
ref = 8
pfa = 1e-6

########################################################

cfars = []
i = 0

details = False
ardPlot = False
cfarPlot = False
cfarCombinedPlot = False
deleteFile = False
save = False
streamRemoteIn = False
streamLocalIn = False
streamOut = True

########################################################

while True:
    if streamRemoteIn:
        try:
            ard = streamRemoteARD(IP, port_r)
        except Exception as e:
            print(e)

    elif streamLocalIn:
        try:
            ard = streamLocalARD(IP, port_r)
        except Exception as e:
            print(e)
        
    else:
        # Find latest file in folder
        latestARD = findARD(path, i)
        i = i + 1

        # Read file
        ard = readARD(latestARD)

        # Delete file after reading it
        if deleteFile:
            os.remove(latestARD)
    
    if details:
        print('--------------------------------------------\n')
        print('File Name: %s' % ard.fileName)
        print('File Type: %s' % ard.fileType)
        print('ARD Type: %s' % ard.ARDType)
        print('Timestamp (us): %i' % ard.TimeStamp_us)
        print('Fc (Hz): %i' % ard.Fc_Hz)
        print('Fs (Hz): %i' % ard.Fs_Hz)
        print('BS (Hz): %i' % ard.BW_Hz)
        print('Range Resolution (m): %f' % ard.RangeResolution_m)
        print('Doppler Resolution (Hz): %f' % ard.DopplerResolution_Hz)
        print('Range Bins: %i' % ard.XDimension)
        print('Doppler Bins: %i' % ard.YDimension)
        print('Tx to Rx Distance (m): %f' % ard.TxToRefRxDistance_m)
        print('File Size (B): %i' % ard.FileSize_B)
        print('\nComment: %s' % ard.Comment)
        print('\n--------------------------------------------\n')    

    centroidMap, centroidMap_snr, detections = cfar(ard, guard, ref, pfa)
    
    # Append the detections to a cfar matrix to plot later
    cfars.append(centroidMap)

    if ardPlot:
        # The 0 represents the desired figure number
        plotARD(ard, 0)

    if cfarPlot:
        plotCFAR(ard, centroidMap_snr, 1)
    
    if cfarCombinedPlot:
        # First need to combine the cfars into a single cfar plot
        if np.size(cfars, 0) == 1:
            detTot = cfars[0]
        else:
            # Take the latest (last cfar) and add it to the existing plot
            detTot = detTot + cfars[-1]
            detTot = detTot >= 1
        
        plotCFAR(ard, detTot, 2)

    ard = extractDetections(ard, centroidMap_snr)

    if save:
        saveCSV(name, ard)

    if streamOut:
        writeSocket(ard, IP, port_s)