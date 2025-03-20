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


settingsFile = 'tools/settings.yaml'

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
            data = data[data.columns.drop(list(data.filter(regex='ExCorr')))]
            data.set_index('X', inplace=True)
        
        except :
            
            data = pd.DataFrame(columns=[''])
        
        return data
    
    def createSpectra(self,Data,Runs,Buffer='None',CamCorrection=False) :
        
        if Buffer != 'None' :
            Data = Data.sub(Data[Buffer], axis=0)
        
        if CamCorrection :
            
            with open(settingsFile, 'r') as stream :
                settings = yaml.safe_load(stream)
            file_path = settings['files']['calmodulin']
            try:
                with open(file_path, 'r'):
                    Cam = pd.read_csv(file_path)
                    Cam.set_index('X', inplace=True)
                    for name in Data.columns :
                        scaling = np.average(Data[name].values[0:3] / Cam['Y'].values[0:3])
                        Data[name] -= scaling*Cam['Y']
                        
            except FileNotFoundError:
                print("Calmodulin data does not exist. Skipping tyrosine removal.")
        
        Data = Data.filter(items=Runs)
        
        
        
        return Data
    
    def plotData(self,Data,Labels='',Title='') :
        
        fontsize = 20
        fig, ax = plt.subplots(figsize=(10,8))
        for name in Data.columns :
            plt.plot(Data[name],label=name)
        plt.plot(Data)
        plt.legend(frameon=False, bbox_to_anchor=(1.02, 1), fontsize=fontsize)
        plt.xlabel('Wavelength (nm)',fontsize=fontsize), plt.ylabel('Intensity (au)',fontsize=fontsize)
        plt.title(Title, fontsize=fontsize)
        ax.tick_params(axis='both',which='both',labelsize=fontsize,direction="in")
        ax.minorticks_on()
        plt.show()
        
        return fig


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
        
        self.BufferNames = ['None']
        self.Names = ['']

        self.FoldersLabel = '-------Folders-------'
        self.FilesLabel = '-------Files-------'
        self.settingsFile = settingsFile
        
        with open(settingsFile, 'r') as stream :
            settings = yaml.safe_load(stream)

        if os.path.isdir(settings['folders']['data']) :
            self.cwd = settings['folders']['data']
        else :
            self.cwd = str(Path(os.getcwd()))

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
                settings['folders']['data'] = str(address)
                with open(settingsFile, "w") as outfile:
                    yaml.dump(settings,outfile,default_flow_style=False)

        def newaddress(value):
            go_to_address(folderField.value)
        folderField = ipw.Text(
            value=self.cwd,
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Current Folder'
            )
        folderField.on_submit(newaddress)
                
        def selecting(value) :
            if value['new'] and value['new'] not in [self.FoldersLabel, self.FilesLabel] :
                path = Path(folderField.value)
                newpath = path / value['new']
                if newpath.is_dir():
                    go_to_address(newpath)
                elif newpath.is_file():
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
            self.Buffer.value = 'None'
            self.Runs_Selected.value = []
            self.data = dt.loadData(folderField.value,SelectFile.value)
            self.filename = SelectFile.value
            RunList()
        load_button = ipw.Button(description='Load',layout=Layout(width='10%'))
        load_button.on_click(load)

        def RunList():
            self.Runs = list(self.data.columns.values)
            Runs = [k for k in self.Runs if self.Filter.value in k]
            self.Runs_Selected.options = Runs
            Runs.insert(0,'None')
            self.Buffer.options = Runs

        self.Filter = ipw.Text(
            value='',
            placeholder='Type something',
            description='Filter:',
            style = {'description_width': '150px'},
            disabled=False
        )
        
        def Update_RunList_Clicked(b):
            RunList()
        Update_RunList = ipw.Button(description="Update run list")
        Update_RunList.on_click(Update_RunList_Clicked)

        self.Runs_Selected = ipw.SelectMultiple(
            options='',
            style = {'width': '100px','description_width': '150px'},
            rows=20,
            layout=Layout(width='70%'),
            description='Runs',
            disabled=False
        )

        self.Buffer = ipw.Dropdown(
            options=self.BufferNames,
            value='None',
            layout=Layout(width='70%'),
            description='Buffer',
            style = {'description_width': '150px'},
            disabled=False,
        )
        
        self.CamCorrection = ipw.Checkbox(
            value=False,
            description='Subtract CaM?',
            disabled=False,
            indent=False
)

        def Plot_Clicked(b):
            with out :
                clear_output()
                self.Spectra = dt.createSpectra(self.data,self.Runs_Selected.value,Buffer=self.Buffer.value,CamCorrection=self.CamCorrection.value)
                self.fig = dt.plotData(self.Spectra)
                display(LowLim)
                display(UpLim)
                display(ipw.Box([button_Integrate,SavePlot,SpectraToClipboard]))
            with anout :
                clear_output()
            LowLim.max = max(self.Spectra.index)
            LowLim.min = min(self.Spectra.index)
            LowLim.value = min(self.Spectra.index)
            UpLim.max = max(self.Spectra.index)
            UpLim.min = min(self.Spectra.index)
            UpLim.value = max(self.Spectra.index)
        Plot = ipw.Button(description="Plot")
        Plot.on_click(Plot_Clicked)

        def SpectraToClipboard_Clicked(b):
            DataToSave = self.Spectra
            DataToSave.to_clipboard()
        SpectraToClipboard = ipw.Button(description="Copy Plot Data")
        SpectraToClipboard.on_click(SpectraToClipboard_Clicked)

        def SavePlot_Clicked(b):
            self.fig.savefig(self.filename.replace('.txt','.jpg'),bbox_inches='tight')
        SavePlot = ipw.Button(description="Save Plot")
        SavePlot.on_click(SavePlot_Clicked)

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
        button_Integrate = ipw.Button(description="Integrate")
        button_Integrate.on_click(Integrate)
        
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
        
        def IntegratedToClipboard_Clicked(b):
            DataToSave = at.integratedValues
            DataToSave.to_clipboard()
        IntegratedToClipboard = ipw.Button(description="Copy integrated data")
        IntegratedToClipboard.on_click(IntegratedToClipboard_Clicked)

        display(ipw.HBox([folderField]))
        display(ipw.HBox([SelectFolder,up_button]))
        display(ipw.HBox([SelectFile,load_button]))
        display(ipw.Box([self.Filter,Update_RunList]))
        display(ipw.Box([self.Runs_Selected,Plot]))
        display(ipw.Box([self.Buffer,self.CamCorrection]))
        

        display(out)
        display(anout)

    def get_folder_contents(self,folder):

        'Gets contents of folder, sorting by folder then files, hiding hidden things'
        folder = Path(folder)
        folders = [item.name for item in folder.iterdir() if item.is_dir() and not item.name.startswith('.')]
        files = [item.name for item in folder.iterdir() if item.is_file() and not item.name.startswith('.')]
        return sorted(folders), sorted(files)