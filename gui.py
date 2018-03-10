__author__ = 'Ze Chen, Zelan Xiang, Ziang Li'
import tkinter as tk
import tkFileDialog
import wave
from playsound import playsound

import numpy as np
import matplotlib
import marsyas
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
matplotlib.use('TkAgg')
class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid()
        self.create()
        self.filename = ''
        self.file = ''
        self.X,self.y = self.train()
    def create(self):
        self.frame_up = tk.Frame(self, background='white')
        self.frame_up.pack(side=tk.TOP)
        self.frame_bottom = tk.Frame(self, background='white')
        self.frame_bottom.pack(side=tk.BOTTOM)

        tk.Label(self.frame_up, text='Input your .wav file:', bg='white').grid(row=0, column=0, padx=15, pady=5)

        self.text1 = tk.Text(self.frame_up, width=30, height=3)
        self.text1.grid(row=0, column=1, padx=5, pady=5)

        self.open_button = tk.Button(self.frame_up, text='Open', width=10, height=2, font=('Arial', 13, 'bold'), fg='white', bg='DarkOrange', command=self.readfile).grid(row=0, column=2, padx=5, pady=5)
        self.plot_button = tk.Button(self.frame_up, text='Plot', width=10, height=2, font=('Arial', 13, 'bold'), fg='white', bg='DarkOrange', command=self.plotwave).grid(row=0, column=3, padx=8.3, pady=5)

        figures = Figure(figsize=(6, 3))
        temp1 = figures.add_subplot(111)
        temp1.plot()
        canvas = FigureCanvasTkAgg(figures, self.frame_bottom)
        canvas.get_tk_widget().grid(row=0, padx=10, pady=10)
        canvas.draw()

        self.detect_button = tk.Button(self.frame_bottom, text='Detect', width=10, height=2, font=('Arial', 13, 'bold'), fg='white', bg='DarkOrange', command=self.detect).grid(row=1, column=0, padx=5, pady=5)

    def readfile(self):
        self.filename = tkFileDialog.askopenfilename(initialdir='/Users/leon/Downloads', title='Select file', filetypes=(('Wave files', '*.wav'), ('all files', '*.*')))
        self.text1.delete(0.0, tk.END)
        self.text1.insert(tk.END, self.filename)

    def plotwave(self):
        read_file = wave.open(self.filename, 'rb')
        params = read_file.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        data = read_file.readframes(nframes)
        read_file.close()
        wave_data = np.fromstring(data, dtype=np.int16)
        time = np.arange(0, nframes) * (1.0 / framerate)
        if nchannels == 2:
            wave_data.shape = -1, 2
            wave_data = wave_data.T
            figures = Figure(figsize=(6, 3))
            temp1 = figures.add_subplot(211)
            temp1.plot(time, wave_data[0], color='b')
            temp2 = figures.add_subplot(211)
            temp2.plot(time, wave_data[1], color='g')
            canvas = FigureCanvasTkAgg(figures, self.frame_bottom)
            canvas.get_tk_widget().grid(row=0, padx=10, pady=10)
            canvas.draw()
        if nchannels == 1:
            figures = Figure(figsize=(6, 3))
            temp1 = figures.add_subplot(111)
            temp1.plot(time, wave_data, color='b')
            canvas = FigureCanvasTkAgg(figures, self.frame_bottom)
            canvas.get_tk_widget().grid(row=0, padx=10, pady=10)
            canvas.draw()
        playsound(self.filename)

    def detect(self):
        sent = "Series {+ input_file = \"%s\"-> input: SoundFileSource { filename = /input_file inSamples = 2048}-> Windowing {size = 2048}-> Spectrum -> PowerSpectrum -> Transposer -> MaxArgMax ->Transposer -> selection: Selector {disable = 0}-> CsvSink { filename = \"wode.csv\" }+ done = (input/hasData == false)}" %self.filename
        file=open("wodefile.mrs", "w")
        file.write(sent)
        file.close()
        mng = marsyas.MarSystemManager()

        fnet = mng.create("Series", "featureNetwork")
        result = []
        system = marsyas.system_from_script_file("wodefile.mrs")
        get = system.getControl


        while get("SoundFileSource/input/mrs_bool/hasData").to_bool():
            system.tick()
            result.extend(get("Selector/selection/mrs_realvec/processedData").to_realvec())
            result[-1] *=  44100 / 2048
        sum = 0.0
        for i in range(0,len(result)):
            sum += result[i]
        avg = np.array(sum/len(result)/1000)
        clfs = [SVC()]
        for clf in clfs:
            clf.fit(self.X, self.y)
            y_pred = clf.predict(avg)
            self.play(y_pred[0])
    def play(self, y_pred):
        if y_pred == 0:
                playsound('/Users/leon/Downloads/male.mp3')
        if y_pred == 1:
                playsound('/Users/leon/Downloads/female.mp3')
    def train(self):
        
        meanf0 = [3.58744615385,
                3.8745704698,
                3.48736363636,
                4.28371232877,
                3.99456521739,
                4.47237313433,
                4.37592857143,
                4.31025,
                4.54233333333,
                4.27741176471,
                1.06794545455,
                0.997572413793,
                0.577573426573,
                0.606188976378,
                0.611,
                0.969382550336,
                0.97335,
                0.46305,
                0.614361702128,
                0.951957055215,
                0.424436619718,
                0.856714285714,
                0.9235,
                0.64256557377,
                0.480576923077,
                0.629052631579,
                0.726372093023,
                1.40886075949,
                1.54842857143,
                1.12249315068,
                1.09774528302,
                1.07631325301,
                2.037,
                1.22863291139,
                0.870130434783,
                1.344,
                0.658285714286,
                0.879822222222,
                0.971637583893,
                0.593063829787,
                0.829744186047,
                0.917663793103,
                0.867,
                0.868569767442,
                0.817875,
                1.06363157895,
                0.562264705882,
                1.99478571429,
                0.471205479452,
                0.703733333333,
                3.34962650602,
                0.94085915493,
                2.15314814815,
                1.55547,
                1.93036666667,
                1.26311111111,
                1.87328571429,
                1.28739130435,
                2.19394736842,
                1.7631369863,
                1.42684931507,
                1.40584931507,
                1.41814814815,
                1.27064383562,
                2.5435890411,
                0.694285714286,
                1.06742045455,
                2.23233333333,
                2.68681690141,
                1.22966666667,
                1.22583050847,
                1.6115,
                1.7668,
                1.58146153846,
                2.12404347826,
                0.966381818182,
                1.37381481481,
                1.47895081967,
                1.66691304348,
                1.04341791045,
                1.01591803279,
                1.065,
                1.30576119403,
                1.05913043478,
                0.798575342466,
                0.954815217391,
                1.36256521739,
                1.51647540984,
                0.945763636364,
                1.54533333333,
                1.06473387097,
                1.14778125,
                1.37025,
                0.826107692308,
                1.44238356164,
                0.536454545455,
                1.68891780822,
                1.81013114754,
                1.68791304348,
                1.56115909091
         ]
         
        lab = [0]*50 + [1]*50
        
        X = np.array(meanf0).astype(float)
        X = np.reshape(X,(-1,1))
        y = np.array(lab).astype(int)
        print (len(X), len(y))
        return X,y

def info():
    widow1 = tk.Toplevel()
    widow1.title('Voice detection information')
    label1 = tk.Label(widow1, text='This project is done by Ze Chen, Zelan Xiang, Ziang Li.', background='white')
    label1.pack(expand=1, padx=10, pady=10, side=tk.TOP)
    tk.Button(widow1, text='Close', width=5, height=1, font=('Arial', 13, 'bold'), fg='white', bg='DarkOrange', command=widow1.destroy).pack(side=tk.BOTTOM)

if __name__ == '__main__':
    COUNT_GENRE = [0] * 20
    ROOT = tk.Tk()
    ROOT.title('Voice detection')
    ROOT.geometry('650x550')

    MB = tk.Menu(ROOT)
    FM = tk.Menu(ROOT, tearoff=0)
    FM.add_command(label='Exit', command=ROOT.quit)
    MB.add_cascade(label='Main', menu=FM)

    HE = tk.Menu(MB, tearoff=0)
    HE.add_command(label='Info', command=info)
    MB.add_cascade(label='Help', menu=HE)
    ROOT.config(menu=MB, background='white')

    APP = Application(master=ROOT)
    APP.mainloop()
