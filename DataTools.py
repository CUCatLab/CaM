import numpy as np
import scipy
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import ipywidgets as ipw
from ipywidgets import Button, Layout
from IPython.display import clear_output
import re
from os import listdir
from os.path import isfile, join
import lmfit
from lmfit import model
from lmfit.models import SkewedGaussianModel, GaussianModel, LinearModel, VoigtModel, PolynomialModel

class Data :
    
    def __init__(self) :
        
        self.BackgroundNames = ['None']
        self.Names = ['']
    
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
        
        self.Spectra = pd.DataFrame(data=np.transpose(Spectra),columns=Header)
    
    def Integrate(self, data, xmin, xmax):
        mask = data['X'].isin(range(xmin, xmax+1))
        idata = data[mask]
        integratedValues = list()
        for idx, column in enumerate(idata) :
            if idx > 0 :
                x = idata['X'].values
                y = idata[column].values
                integratedValues.append(scipy.integrate.trapz(y,x=x))
        integratedValues = pd.DataFrame(data=integratedValues,index=self.Spectra.columns[1:],columns=['Integrated'])
        return idata, integratedValues

    def UI(self) :
        
        out = ipw.Output()
        anout = ipw.Output()
            
        def UpdateFileList(b) :
            self.FolderPath.value = self.FolderPath.value.replace('\\','/')
            self.FileName.options = self.DataList(self.FolderPath.value)
        button_UpdateFileList = ipw.Button(description="Update")
        button_UpdateFileList.on_click(UpdateFileList)

        def Update_Runs_Clicked(b):
            Runs = self.Runs()
            Runs = [k for k in Runs if self.Filter.value in k]
            self.Runs_Selected.options = Runs
            Runs.insert(0,'None')
            self.Background.options = Runs
        Update_Runs = ipw.Button(description="Update run list")
        Update_Runs.on_click(Update_Runs_Clicked)

        def SpectraToClipboard_Clicked(b):
            DataToSave = self.Spectra
            DataToSave.to_clipboard()
        SpectraToClipboard = ipw.Button(description="Copy plot data")
        SpectraToClipboard.on_click(SpectraToClipboard_Clicked)
        
        def IntegratedToClipboard_Clicked(b):
            DataToSave = self.integratedValues
            DataToSave.to_clipboard()
        IntegratedToClipboard = ipw.Button(description="Copy integrated data")
        IntegratedToClipboard.on_click(IntegratedToClipboard_Clicked)

        def Update_Plot_Clicked(b):
            with out :
                clear_output(True)
                self.Plot(self.Runs_Selected.value,Background=self.Background.value)
            LowLim.max = max(self.Spectra['X'].values)
            LowLim.min = min(self.Spectra['X'].values)
            LowLim.value = min(self.Spectra['X'].values)
            UpLim.max = max(self.Spectra['X'].values)
            UpLim.min = min(self.Spectra['X'].values)
            UpLim.value = max(self.Spectra['X'].values)
        Update_Plot = ipw.Button(description="Plot")
        Update_Plot.on_click(Update_Plot_Clicked)

        def LoadData(b):
            with out :
                clear_output(True)
            self.Data = self.LoadData(self.FileName.value)
            Runs = self.Runs()
            self.Runs_Selected.options = Runs
            Runs.insert(0,'None')
            self.Background.options = Runs
        button_LoadData = ipw.Button(description="Load data")
        button_LoadData.on_click(LoadData)

        def Integrate(b):
            with anout :
                clear_output(True)
                self.idata, self.integratedValues = self.Integrate(self.Spectra, LowLim.value, UpLim.value)
                self.idata = pd.DataFrame(index=self.Spectra.columns)
                plt.figure(figsize=(13,7))
                plt.xlabel('Run'), plt.ylabel('Integrated Value')
                plt.plot(self.integratedValues, '.-')
                plt.show()
                display(IntegratedToClipboard)
        button_Integrate = ipw.Button(description="Integrate")
        button_Integrate.on_click(Integrate)

        self.FolderPath = ipw.Text(
            value='../',
            placeholder='Type file path',
            description='Folder',
            layout=Layout(width='80%'),
            style = {'description_width': '150px'},
            disabled=False
        )

        self.FileName = ipw.Dropdown(
            options=self.DataList(self.FolderPath.value),
            description='File',
            layout=Layout(width='80%'),
            style = {'description_width': '150px'},
            disabled=False,
        )

        self.Filter = ipw.Text(
            value='',
            placeholder='Type something',
            description='Filter:',
            style = {'description_width': '150px'},
            disabled=False
        )

        self.Background = ipw.Dropdown(
            options=self.BackgroundNames,
            value='None',
            layout=Layout(width='80%'),
            description='Select background run',
            style = {'description_width': '150px'},
            disabled=False,
        )

        self.Runs_Selected = ipw.SelectMultiple(
            options=self.Names,
            style = {'width': '100px','description_width': '150px'},
            rows=20,
            layout=Layout(width='80%'),
            description='Select runs',
            disabled=False
        )
        
        LowLim = ipw.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description='Lower Limit:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
            )

        UpLim = ipw.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description='Upper Limit:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
            )

        display(ipw.Box([self.FolderPath,button_UpdateFileList]))
        display(ipw.Box([self.FileName,button_LoadData]))
        display(ipw.Box([self.Filter,Update_Runs]))
        display(self.Runs_Selected)
        display(self.Background)
        display(ipw.Box([Update_Plot,SpectraToClipboard]))
        display(LowLim)
        display(UpLim)
        display(ipw.Box([button_Integrate]))

        display(out)
        display(anout)