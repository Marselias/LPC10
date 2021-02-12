import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import struct
file = r'mowa.wav'
data, samplerate = sf.read(file)
frame_len = 256
prediction_order = 10
bits = 3

def segmentation(file2):
    segments = []
    n = len(file2)
    n = int(n/frame_len)
    for i in range(n):
        segment = file2[i*frame_len:(i+1)*frame_len]
        segments.append(segment)
    return segments

def pre_emphasis(file):
    alpha = 0.9735
    pre_emphasis2 = [file[0]]
    for i in range(1, len(file)):
        pre_emphasis2.append(file[i]-alpha*file[i-1])
    return pre_emphasis2

def windowing(file):
    window = np.hanning(len(file))
    for i in range(len(file)):
        window[i] = window[i]*file[i]
    return window

def autocorr(file):
    result = np.correlate(file, file, mode='full')
    result = result[int(len(result) / 2):]
    return result
def autocorr_biased(file):
    c=[]
    N=len(file)
    for i in range(len(file)):
        c.append(file[i]/N)
    return c

def autocorr_normalized(file):
    c=[]
    top = max(file)
    for i in range(len(file)):
        c.append(file[i]/top)
    return c

def levinson_durbin1(R):
    a = np.ndarray(prediction_order+1)
    a[0] = 1
    K = -R[1]/R[0]
    a[1] = K
    Alpha = R[0]*(1-K*K)
    a_new=a.copy()
    for i in range(2,prediction_order+1):
        suma = 0
        for j in range(1,i):
            suma = suma + R[j]*a[i-j]
        suma = suma + R[i]
        K = -suma/Alpha
        for j in range(1,i):
            a_new[j] = a[j] + K * a[i-j]
        a_new[i] = K
        a=a_new.copy()
        Alpha = Alpha*(1-K*K)
    return a_new


def errors(file10, previous10, a):
    c = np.array(previous10)
    previous10 = np.concatenate((c,file10))
    errors = []
    y_hat = []
    for i in range(10, len(previous10)):
        prediction = (a[0]*previous10[i-1] + a[1]*previous10[i-2] + a[2]*previous10[i-3] + a[3]*previous10[i-4] + a[4]*previous10[i-5] + a[5]*previous10[i-6] + a[6]*previous10[i-7] + a[7]*previous10[i-8] + a[8]*previous10[i-9] + a[9]*previous10[i-10])
        errors.append(previous10[i]+prediction)
        y_hat.append(prediction)
    return y_hat, errors

def quantization(errors,bits):
    emax = max(errors)
    emin = min(errors)
    N = 2 ** bits
    delta = (emax - emin)/N

    quantized=[]
    for e in errors:
        if(e >= emin - 0.01 and e < (emin + delta)):
            quantized.append(0)
        elif (e>=(emin+(N-1)*delta) and e<=(emin+N*delta)+0.01):
            quantized.append(N-1)
        else:
            for i in range(1,N-1):
                if (e>=emin+i*delta and e<(emin+(i+1)*delta)):
                    quantized.append(i)


    return quantized

def stream(emax,emin,quantized,a,bits=0):

    ema = [emax]
    emi = [emin]
    file = open('3bity.txt','a+',encoding='utf-8')
    code = np.concatenate((ema,emi,a))
    for c in code:
        ba = bytearray(struct.pack("f", c))
        for b in ba:
            to_write = chr(b)
            file.write(to_write)
    buf=''
    for q in quantized:
        to_write = format(int(q), '#0%db' % (2+bits))
        to_write = to_write[2:]
        for i in range(len(to_write)):
            buf = buf + to_write[i]
            if len(buf)==8:
                buf = int(buf,2)
                buf = chr(buf)
                file.write(buf)
                buf=''
    file.close()
if __name__=='__main__':
    errors_=[]
    yhat = []
    segmentated = segmentation(data)
    for i in range(1, len(segmentated)):
        wind = windowing(segmentated[i])
        R=autocorr_normalized(autocorr(wind))
        a = levinson_durbin1(R)
        last10=segmentated[i-1][-10:]
        y_hat,e = errors(segmentated[i],last10,a[1:])
        quant = quantization(e,bits)
        errors_.extend(e)
        yhat.extend(y_hat)
        stream(max(e),min(e),quant,a[1:],bits)

    plt.subplot(311)
    plt.plot(data)
    plt.subplot(312)
    plt.plot(yhat)
    plt.subplot(313)
    plt.plot(errors_)
    plt.show()


