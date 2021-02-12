import soundfile as sf
import numpy as np
import struct

def decoder():
    file = open(r'C:\Users\MSI\Desktop\LPC10\2bity.txt', 'r', encoding='utf-8')
    sound = []
    length = 1819
    nbits = 2
    a_format = 4
    to_read = a_format*12 + 256/(8/nbits)
    a=np.ndarray(10)
    for i in range(length):
        line = file.read(int(to_read))
        b=[]

        for i in range(12):
            buf = []
            for j in range(a_format):
                buf.append(ord(line[j+4*i]))
            aa = bytearray(buf)
            b.append(struct.unpack('<f',aa))
        emax = b[0][0]
        emin = b[1][0]
        for i in range(10):
            a[i] = b[i+2][0]
        quant = decode(line[a_format*12:],nbits)
        dequant = dequantization(emin,emax,quant,nbits)
        y = syntesize(dequant,a)
        sound.extend(y)
    file.close()
    sf.write('mowa_zsyntezowana4.wav',sound,8000)


def decode(data,nbits):
    quantized = []
    for d in data:
        sample = format(ord(d), '#010b')
        for i in range(int(8/nbits)):
            quantized.append(int(sample[i*2+2:(i+1)*2+2],2))
    return quantized


def dequantization(emin,emax,quant,bits):
    delta = (emax - emin)/(2**bits)
    dequant = []
    for q in quant:
        dequant.append(emin + delta/2 + delta*q)
    return dequant


def syntesize(e,a):
    bs = np.zeros((10,1))
    y = []
    for i in range(len(e)):
        suma = 0
        for j in range(len(a)):
            suma = suma + a[j]*bs[j]
        out = e[i] -suma
        y.append(out)
        bs = np.concatenate((np.array([out]),bs[:len(bs)-1]))
    return y

decoder()