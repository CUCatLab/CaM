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
