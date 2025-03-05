import numpy as np
import scipy
from scipy import integrate
import pandas as pd
from pandas import DataFrame as df
import yaml
import matplotlib.pyplot as plt
import ipywidgets as ipw
from ipywidgets import Button, Layout
from IPython.display import clear_output
from IPython.display import display_html
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import csv


class dataTools :

    def __init__(self) :

        pass

    def loadData(self,folder,file) :
        
        filepath = folder + '/' + file
        
        try :
            with open(filepath) as f:
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
            data = np.zeros((DataSets,DataEnd[0]-DataStart[0]+1))
            for index in range(len(DataStart)):
                for x in range(DataLength[DataStart[index]]):
                    Header.append(Content[DataStart[index]-2].split('\t')[x])
                    for y in range(DataEnd[0]-DataStart[0]+1):
                        if Content[DataStart[index]+y].split('\t')[x] != '' :
                            data[x + index * DataLength[DataStart[index-1]]][y] = Content[DataStart[index]+y].split('\t')[x]
            for index in range(2,int(len(data)/2)+1):
                data = np.delete(data,index,axis=0)
                Header.remove(Header[index-1])
            Header.remove(Header[-1])
            Header.insert(0,'X')
            data = pd.DataFrame(data=np.transpose(data),columns=Header)
        
        except :
            
            data = pd.DataFrame(columns=[''])
        
        return data
    
    def plotData(self,Data,Runs,Background='None',Labels='',Title='') :
        
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
        
        fontsize = 20
        fig, ax = plt.subplots(figsize=(10,8))
        for i in range (Spectra.shape[0] - 1):
            plt.plot(Spectra[0],Spectra[i+1],label=Labels[i])
        plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.5, 1), ncol=1, fontsize=fontsize)
        plt.xlabel('Wavelength (nm)',fontsize=fontsize), plt.ylabel('Intensity (au)',fontsize=fontsize)
        plt.title(Title, fontsize=fontsize)
        ax.tick_params(axis='both',which='both',labelsize=fontsize,direction="in")
        ax.minorticks_on()
        plt.show()
        
        Header = list(Runs)
        Header.insert(0,'X')
        
        Spectra = pd.DataFrame(data=np.transpose(Spectra),columns=Header)

        return Spectra, fig


class analysisTools :

    def __init__(self) :

        pass

    def Integrate(self, data, xmin, xmax):
        mask = data['X'].isin(range(xmin, xmax+1))
        idata = data[mask]
        integratedValues = list()
        for idx, column in enumerate(idata) :
            if idx > 0 :
                x = idata['X'].values
                y = idata[column].values
                integratedValues.append(integrate.trapezoid(y,x=x))
        integratedValues = pd.DataFrame(data=integratedValues,index=data.columns[1:],columns=['Integrated'])
        return idata, integratedValues


class UI :
    
    def __init__(self) :

        dt = dataTools()
        at = analysisTools()
        
        self.BackgroundNames = ['None']
        self.Names = ['']

        self.cwd = Path(os.getcwd())

        self.FoldersLabel = '-------Folders-------'
        self.FilesLabel = '-------Files-------'

        out = ipw.Output()
        anout = ipw.Output()

        def go_to_address(address):
            address = Path(address)
            if address.is_dir():
                folderField.value = str(address)
                SelectFolder.unobserve(selecting, names='value')
                SelectFolder.options = self.get_folder_contents(folder=address)[0]
                SelectFolder.observe(selecting, names='value')
                SelectFolder.value = None
                SelectFile.options = self.get_folder_contents(folder=address)[1]

        def newaddress(value):
            go_to_address(folderField.value)
        folderField = ipw.Text(value=str(self.cwd),
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Current Folder')
        folderField.on_submit(newaddress)
                
        def selecting(value) :
            if value['new'] and value['new'] not in [self.FoldersLabel, self.FilesLabel] :
                path = Path(folderField.value)
                newpath = path / value['new']
                if newpath.is_dir():
                    go_to_address(newpath)
                elif newpath.is_file():
                    #some other condition
                    pass
        
        SelectFolder = ipw.Select(
            options=self.get_folder_contents(self.cwd)[0],
            rows=5,
            value=None,
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Subfolders')
        SelectFolder.observe(selecting, names='value')
        
        SelectFile = ipw.Select(
            options=self.get_folder_contents(self.cwd)[1],
            rows=10,
            values=None,
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Files')

        def parent(value):
            new = Path(folderField.value).parent
            go_to_address(new)
        up_button = ipw.Button(description='Up',layout=Layout(width='10%'))
        up_button.on_click(parent)
            
        def load(b):
            with out :
                clear_output()
            with anout :
                clear_output()
            self.Background.value = 'None'
            self.Runs_Selected.value = []
            self.data = dt.loadData(folderField.value,SelectFile.value)
            self.filename = SelectFile.value
            RunList()
        load_button = ipw.Button(description='Load',layout=Layout(width='10%'))
        load_button.on_click(load)

        def RunList():
            self.Runs = list(self.data.columns.values)
            self.Runs.remove(self.Runs[0])
            Runs = [k for k in self.Runs if self.Filter.value in k]
            self.Runs_Selected.options = Runs
            Runs.insert(0,'None')
            self.Background.options = Runs
        
        def Update_RunList_Clicked(b):
            RunList()
        Update_RunList = ipw.Button(description="Update run list")
        Update_RunList.on_click(Update_RunList_Clicked)

        def SpectraToClipboard_Clicked(b):
            DataToSave = self.Spectra
            DataToSave.to_clipboard()
        SpectraToClipboard = ipw.Button(description="Copy Plot Data")
        SpectraToClipboard.on_click(SpectraToClipboard_Clicked)
        
        def IntegratedToClipboard_Clicked(b):
            DataToSave = at.integratedValues
            DataToSave.to_clipboard()
        IntegratedToClipboard = ipw.Button(description="Copy integrated data")
        IntegratedToClipboard.on_click(IntegratedToClipboard_Clicked)

        def Plot_Clicked(b):
            with out :
                clear_output()
                self.Spectra, self.fig = dt.plotData(self.data,self.Runs_Selected.value,Background=self.Background.value)
            with anout :
                clear_output()
            LowLim.max = max(self.Spectra['X'].values)
            LowLim.min = min(self.Spectra['X'].values)
            LowLim.value = min(self.Spectra['X'].values)
            UpLim.max = max(self.Spectra['X'].values)
            UpLim.min = min(self.Spectra['X'].values)
            UpLim.value = max(self.Spectra['X'].values)
        Plot = ipw.Button(description="Plot")
        Plot.on_click(Plot_Clicked)

        def Integrate(b):
            with anout :
                clear_output()
                self.idata, self.integratedValues = at.Integrate(self.Spectra, LowLim.value, UpLim.value)
                self.idata = pd.DataFrame(index=self.Spectra.columns)
                plt.figure(figsize=(13,7))
                plt.xlabel('Run',fontsize=16), plt.ylabel('Integrated Value',fontsize=16)
                plt.plot(self.integratedValues, '.-')
                plt.tick_params(axis="x", labelsize=16, rotation=-90)
                plt.tick_params(axis="y", labelsize=16)
                plt.show()
                display(IntegratedToClipboard)
        button_Integrate = ipw.Button(description="Integrate")
        button_Integrate.on_click(Integrate)

        def SavePlot_Clicked(b):
            self.fig.savefig(self.filename.replace('.txt','.jpg'),bbox_inches='tight')
        SavePlot = ipw.Button(description="Save Plot")
        SavePlot.on_click(SavePlot_Clicked)

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
            description='Background Run',
            style = {'description_width': '150px'},
            disabled=False,
        )

        self.Runs_Selected = ipw.SelectMultiple(
            options=self.Names,
            style = {'width': '100px','description_width': '150px'},
            rows=20,
            layout=Layout(width='80%'),
            description='Runs',
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

        display(ipw.HBox([folderField]))
        display(ipw.HBox([SelectFolder,up_button]))
        display(ipw.HBox([SelectFile,load_button]))
        display(ipw.Box([self.Filter,Update_RunList]))
        display(self.Runs_Selected)
        display(self.Background)
        display(ipw.Box([Plot,SpectraToClipboard]))
        display(LowLim)
        display(UpLim)
        display(ipw.Box([button_Integrate,SavePlot]))

        display(out)
        display(anout)

    def get_folder_contents(self,folder):

        'Gets contents of folder, sorting by folder then files, hiding hidden things'
        folder = Path(folder)
        folders = [item.name for item in folder.iterdir() if item.is_dir() and not item.name.startswith('.')]
        files = [item.name for item in folder.iterdir() if item.is_file() and not item.name.startswith('.')]
        return sorted(folders), sorted(files)