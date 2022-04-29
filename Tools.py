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

class Fluorescence :
    
    def __init__(self) :
        
        self.BackgroundNames = ['None']
        self.Names = ['']

        self.cwd = Path(os.getcwd())

        self.FoldersLabel = '-------Folders-------'
        self.FilesLabel = '-------Files-------'

    def get_folder_contents(self,folder):

        'Gets contents of folder, sorting by folder then files, hiding hidden things'
        folder = Path(folder)
        folders = [item.name for item in folder.iterdir() if item.is_dir() and not item.name.startswith('.')]
        files = [item.name for item in folder.iterdir() if item.is_file() and not item.name.startswith('.')]
        return sorted(folders), sorted(files)
    
    def LoadData(self,folder,file) :
        
        self.filename = file
        self.foldername = folder
        self.filepath = folder + '/' + file
        
        try :
            with open(self.filepath) as f:
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
        
        self.Spectra = pd.DataFrame(data=np.transpose(Spectra),columns=Header)
        self.fig = fig
    
    def Integrate(self, data, xmin, xmax):
        mask = data['X'].isin(range(xmin, xmax+1))
        idata = data[mask]
        integratedValues = list()
        for idx, column in enumerate(idata) :
            if idx > 0 :
                x = idata['X'].values
                y = idata[column].values
                integratedValues.append(integrate.trapz(y,x=x))
        integratedValues = pd.DataFrame(data=integratedValues,index=self.Spectra.columns[1:],columns=['Integrated'])
        return idata, integratedValues

    def UI(self) :
        
        out = ipw.Output()
        anout = ipw.Output()

        def go_to_address(address):
            address = Path(address)
            if address.is_dir():
                address_field.value = str(address)
                SelectFolder.unobserve(selecting, names='value')
                SelectFolder.options = self.get_folder_contents(folder=address)[0]
                SelectFolder.observe(selecting, names='value')
                SelectFolder.value = None
                SelectFile.options = self.get_folder_contents(folder=address)[1]

        def newaddress(value):
            go_to_address(address_field.value)
        address_field = ipw.Text(value=str(self.cwd),
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Current Folder')
        address_field.on_submit(newaddress)
                
        def selecting(value) :
            if value['new'] and value['new'] not in [self.FoldersLabel, self.FilesLabel] :
                path = Path(address_field.value)
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
            new = Path(address_field.value).parent
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
            self.LoadData(address_field.value,SelectFile.value)
            Runs = self.Runs()
            self.Runs_Selected.options = Runs
            Runs.insert(0,'None')
            self.Background.options = Runs
        load_button = ipw.Button(description='Load',layout=Layout(width='10%'))
        load_button.on_click(load)

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
        SpectraToClipboard = ipw.Button(description="Copy Plot Data")
        SpectraToClipboard.on_click(SpectraToClipboard_Clicked)
        
        def IntegratedToClipboard_Clicked(b):
            DataToSave = self.integratedValues
            DataToSave.to_clipboard()
        IntegratedToClipboard = ipw.Button(description="Copy integrated data")
        IntegratedToClipboard.on_click(IntegratedToClipboard_Clicked)

        def Plot_Clicked(b):
            with out :
                clear_output()
                self.Plot(self.Runs_Selected.value,Background=self.Background.value)
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
                self.idata, self.integratedValues = self.Integrate(self.Spectra, LowLim.value, UpLim.value)
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

        display(ipw.HBox([address_field]))
        display(ipw.HBox([SelectFolder,up_button]))
        display(ipw.HBox([SelectFile,load_button]))
        display(ipw.Box([self.Filter,Update_Runs]))
        display(self.Runs_Selected)
        display(self.Background)
        display(ipw.Box([Plot,SpectraToClipboard]))
        display(LowLim)
        display(UpLim)
        display(ipw.Box([button_Integrate,SavePlot]))

        display(out)
        display(anout)


class CD :
    
    def __init__(self) :
        
        self.BackgroundNames = ['None']
        self.Names = ['']

        self.cwd = Path(os.getcwd())

        self.FoldersLabel = '-------Folders-------'
        self.FilesLabel = '-------Files-------'

    def get_folder_contents(self,folder):

        'Gets contents of folder, sorting by folder then files, hiding hidden things'
        folder = Path(folder)
        folders = [item.name for item in folder.iterdir() if item.is_dir() and not item.name.startswith('.')]
        files = [item.name for item in folder.iterdir() if item.is_file() and not item.name.startswith('.')]
        return sorted(folders), sorted(files)
    
    def LoadData(self,folder,files) :
        
        self.files = files
        self.folder = folder
        self.Data = list()

        try :
            for file in files :
                filepath = folder + '/' + file
                with open(filepath) as f:
                    Content = f.readlines()
                    Start = Content.index('XYDATA\n') + 1
                    End = Content.index('##### Extended Information\n') - 1
                    Content = Content[Start:End]
                    reader = csv.reader(Content)
                    df = pd.DataFrame(reader)
                    df = df.astype('float')
                    header = list()
                    for i in range(df.shape[1]) :
                        if i == 0 :
                            header.append('x')
                        else :
                            header.append('y'+str(i))
                    df.columns = header
                    self.Data.append(df)
        except :
            pass
        
        return self.Data
    
    def Plot(self,Data,Channel=1,Labels='',Title='') :
        
        Spectra=pd.DataFrame()
        fontsize = 20
        fig, ax = plt.subplots(figsize=(10,8))
        for i in range(len(Data)) :
            ylabel = Data[i].columns[Channel]
            if Labels == '' :
                Label = self.files[i]
            else:
                Label = Label[i]
            plt.plot(Data[i]['x'],Data[i][ylabel],label=Label)
            if i == 0 :
                Spectra['x'] = Data[i]['x']
            Spectra[Label] = Data[i][ylabel]
        plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.5, 1), ncol=1, fontsize=fontsize)
        plt.xlabel('Wavelength (nm)',fontsize=fontsize), plt.ylabel('Intensity (au)',fontsize=fontsize)
        plt.title(Title, fontsize=fontsize)
        ax.tick_params(axis='both',which='both',labelsize=fontsize,direction="in")
        ax.minorticks_on()
        plt.show()
        
        self.Spectra = Spectra
        self.fig = fig

    def UI(self) :
        
        out = ipw.Output()
        anout = ipw.Output()

        def go_to_address(address):
            address = Path(address)
            if address.is_dir():
                address_field.value = str(address)
                SelectFolder.unobserve(selecting, names='value')
                SelectFolder.options = self.get_folder_contents(folder=address)[0]
                SelectFolder.observe(selecting, names='value')
                SelectFolder.value = None
                SelectFiles.options = self.get_folder_contents(folder=address)[1]

        def newaddress(value):
            go_to_address(address_field.value)
        address_field = ipw.Text(value=str(self.cwd),
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Current Folder')
        address_field.on_submit(newaddress)
                
        def selecting(value) :
            if value['new'] and value['new'] not in [self.FoldersLabel, self.FilesLabel] :
                path = Path(address_field.value)
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
        
        SelectFiles = ipw.SelectMultiple(
            options=self.get_folder_contents(self.cwd)[1],
            rows=10,
            values=None,
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Files')

        def parent(value):
            newpath = Path(address_field.value).parent
            go_to_address(newpath)
            SelectFiles.options = self.get_folder_contents(newpath)[1]
        up_button = ipw.Button(description='Up',layout=Layout(width='10%'))
        up_button.on_click(parent)
            
        def load(b):
            with out :
                clear_output()
            with anout :
                clear_output()
            self.LoadData(address_field.value,SelectFiles.value)
            with out :
                clear_output()
                self.Plot(self.Data,Channel=Channel.value)
                display(Plot2Clipboard)
            with anout :
                clear_output()
        load_button = ipw.Button(description='Load',layout=Layout(width='10%'))
        load_button.on_click(load)

        Channel = ipw.IntText(
            value=1,
            description='Channel:',
            layout=Layout(width='15%'),
            style = {'width': '100px','description_width': '150px'},
            disabled=False
        )

        def Pot2Clipboard_Clicked(b):
            DataToSave = self.Spectra
            DataToSave.to_clipboard()
        Plot2Clipboard = ipw.Button(description="Copy Spectra")
        Plot2Clipboard.on_click(Pot2Clipboard_Clicked)

        display(ipw.HBox([address_field]))
        display(ipw.HBox([SelectFolder,up_button]))
        display(ipw.HBox([SelectFiles,load_button]))
        display(Channel)

        display(out)
        display(anout)


class Calculations :
    
    def __init__(self) :
        
        with open('Molecules.yaml', 'r') as stream:
            self.Molecules = yaml.safe_load(stream)
        
        Molecules = self.Molecules
        
        MoleculesTable = pd.DataFrame()
        Names = list()
        Mass = list()
        
        for molecule in Molecules['Proteins'] :
            Names.append(molecule)
            Mass.append(self.ProteinMass(Molecules['Proteins'][molecule]['Sequence'])[0])
        
        for molecule in Molecules['Solutes'] :
            Names.append(molecule)
            Mass.append(Molecules['Solutes'][molecule]['Mass'])
        
        MoleculesTable.index = Names
        MoleculesTable['Mass (g/mol)'] = Mass
        MoleculesTable['Mass (g/mol)'] = MoleculesTable['Mass (g/mol)'].round(decimals=1)
        
        self.MoleculesTable = MoleculesTable
    
    def ProteinMass(self, seq) :
        if '{LYS(FITC)}' in seq :
            mass = 389.38
            seq = seq.replace('{LYS(FITC)}','')
        else:
            mass = 0
        masss = {'A': 71.04, 'C': 103.01, 'D': 115.03, 'E': 129.04, 'F': 147.07,
               'G': 57.02, 'H': 137.06, 'I': 113.08, 'K': 128.09, 'L': 113.08,
               'M': 131.04, 'N': 114.04, 'P': 97.05, 'Q': 128.06, 'R': 156.10,
               'S': 87.03, 'T': 101.05, 'V': 99.07, 'W': 186.08, 'Y': 163.06 }
        mass += sum(masss[p] for p in seq)
        return mass, "Daltons"
    
    def MoleculesTable(self) :
        
        pass
    
    def g2Add(self, m, M=50, V=1.0) :
        # m molecular mass in units of g/mol (Daltons)
        # M molarity in units of micromolar
        # V volumen in units of mL
        M = M/1e6
        V = V/1e3
        m = m*1000
        return M*V*m, 'mg'
    
    def MassCalculator(self) :
        
        out = ipw.Output()
        
        Sequence = ipw.Text(
            value='',
            placeholder='Paste sequence',
            description='Sequence:',
            layout=Layout(width='75%'),
            disabled=False
        )
        
        def Calculate(b) :
            with out :
                clear_output()
                mass = self.ProteinMass(Sequence.value)
                print("The molecular mass of this protein is", f'{mass[0]:.1f}', mass[1])
        button_Calculate = ipw.Button(description="Calculate")
        button_Calculate.on_click(Calculate)
        
        display(ipw.Box([Sequence,button_Calculate]))
        display(out)
    
    def SolutionCalculator(self) :
        
        out = ipw.Output()
        
        Volume = ipw.widgets.FloatText(
            value=1.0,
            description='Volume (mL):',
            style = {'description_width': '120px'},
            disabled=False
        )
        
        Molarity = ipw.widgets.FloatText(
            value=50.0,
            description='Concentration (μM):',
            style = {'description_width': '120px'},
            disabled=False
        )
        
        Molecule = ipw.Dropdown(
            options=self.MoleculesTable.index,
            value=self.MoleculesTable.index[0],
            description='Number:',
            style = {'description_width': '120px'},
            disabled=False,
        )
        
        def Calculate(b) :
            with out :
                clear_output()
                mass = self.g2Add(self.MoleculesTable['Mass (g/mol)'].loc[Molecule.value],Molarity.value,Volume.value)
                print(f'{mass[0]:.3f}', mass[1])
        button_Calculate = ipw.Button(description="Calculate")
        button_Calculate.on_click(Calculate)
        
        display(Volume)
        display(Molarity)
        display(Molecule)
        display(button_Calculate)
        display(out)

# class FileLoader :
    
#     def __init__(self) :

#         cwd = Path(os.getcwd())

#         self.FoldersLabel = '-------Folders-------'
#         self.FilesLabel = '-------Files-------'

#     def get_folder_contents(self,folder):
#         'Gets contents of folder, sorting by folder then files, hiding hidden things'
#         folder = Path(folder)
#         folders = [item.name for item in folder.iterdir() if item.is_dir() and not item.name.startswith('.')]
#         files = [item.name for item in folder.iterdir() if item.is_file() and not item.name.startswith('.')]
#         return [self.FoldersLabel] + sorted(folders) + [self.FilesLabel] + sorted(files)

#     def UI(self):

        