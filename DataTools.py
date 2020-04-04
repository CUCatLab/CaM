import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import ipywidgets as widgets
import re
from os import listdir
from os.path import isfile, join


class Data :
    
    def __init__(self) :
        
        pass
    
    def DataList(self, Path) :
        
        self.Path = Path
        DataList = [f for f in listdir(self.Path) if isfile(join(self.Path, f))]
        return DataList
    
    def LoadData(self, File) :
        
        self.File = File
        
        try :
            with open(self.Path+"/"+self.File) as f:
                Content = f.readlines()
            DataLength = list()
            for index in range(len(Content)):
                DataLength.append(len(Content[index].split('\t')))
            DataStart = list()
            DataEnd = list()
            Counter = 0
            for index in range(len(DataLength)):
                if DataLength[index] == 1 :
                    if Counter > 1 : DataEnd.append(index-1)
                    Counter = 0
                else :
                    if Counter == 0 : DataStart.append(index+3)
                    Counter = Counter + 1
            Header = list()
            DataSets = 0
            for index in range(len(DataStart)):
                DataSets = DataSets + DataLength[DataStart[index]]
            self.Data = np.zeros((DataSets,DataEnd[0]-DataStart[0]+1))
            for index in range(len(DataStart)):
                for x in range(DataLength[DataStart[index]]):
                    Header.append(Content[DataStart[index]-2].split('\t')[x])
                    for y in range(DataEnd[0]-DataStart[0]+1):
                        if Content[DataStart[index]+y].split('\t')[x] != '' :
                            self.Data[x + index * DataLength[DataStart[index-1]]][y] = Content[DataStart[index]+y].split('\t')[x]
            for index in range(2,int(len(self.Data)/2)+1):
                self.Data = np.delete(self.Data,index,axis=0)
                Header.remove(Header[index-1])
            Header.remove(Header[-1])
            Header.insert(0,'X')
            self.Data = pd.DataFrame(data=np.transpose(self.Data),columns=Header)
        
        except :
            
            self.Data = pd.DataFrame(columns=[''])
        
        return self.Data
    
    def Runs(self) :
        
        Runs = list(self.Data.columns.values)
        Runs.remove(Runs[0])
        return Runs
    
    def Plot(self,Runs,Background='None',Labels='',Title='') :
        
        Data = self.Data
        if Background != 'None' :
            Buffer = Data[Background]
        if Labels == '' :
            Labels = Runs
            
        Spectra = np.zeros((len(Runs)+1,len(Data[Data.columns[0]])))
        Spectra[0] = Data[Data.columns[0]]
        i = 0
        for i in range (len(Runs)) :
            Spectra[i+1] = Data[Runs[i]]
            if Background != 'None' :
                Spectra[i+1] = Spectra[i+1] - Buffer
        
        plt.figure(figsize=(8,8))
        plt.xlabel('Wavelength (nm)',fontsize=16), plt.ylabel('Intensity (au)',fontsize=16)
        for i in range (Spectra.shape[0] - 1):
            plt.plot(Spectra[0],Spectra[i+1],label=Labels[i])
        plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.5, 1), ncol=1, fontsize=16)
        plt.title(Title, fontsize=16)
        plt.tick_params(axis="x", labelsize=16)
        plt.tick_params(axis="y", labelsize=16)
        plt.show()
        
        Header = list(Runs)
        Header.insert(0,'X')
        Spectra = pd.DataFrame(data=np.transpose(Spectra),columns=Header)
        self.Spectra = Spectra