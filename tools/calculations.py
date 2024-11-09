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


def ProteinMass(seq) :
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

def g2Add(m, M=50, V=1.0) :
    # m molecular mass in units of g/mol (Daltons)
    # M molarity in units of micromolar
    # V volumen in units of mL
    M = M/1e6
    V = V/1e3
    m = m*1000
    return M*V*m, 'mg'

def MassCalculator() :
    
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
            mass = ProteinMass(Sequence.value)
            print("The molecular mass of this protein is", f'{mass[0]:.1f}', mass[1])
    button_Calculate = ipw.Button(description="Calculate")
    button_Calculate.on_click(Calculate)
    
    display(ipw.Box([Sequence,button_Calculate]))
    display(out)

def SolutionCalculator() :
    
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
        options=MoleculesTable.index,
        value=MoleculesTable.index[0],
        description='Number:',
        style = {'description_width': '120px'},
        disabled=False,
    )
    
    def Calculate(b) :
        with out :
            clear_output()
            mass = g2Add(MoleculesTable['Mass (g/mol)'].loc[Molecule.value],Molarity.value,Volume.value)
            print(f'{mass[0]:.3f}', mass[1])
    button_Calculate = ipw.Button(description="Calculate")
    button_Calculate.on_click(Calculate)
    
    display(Volume)
    display(Molarity)
    display(Molecule)
    display(button_Calculate)
    display(out)

with open('tools/Molecules.yaml', 'r') as stream:
    Molecules = yaml.safe_load(stream)

Molecules = Molecules

MoleculesTable = pd.DataFrame()
Names = list()
Mass = list()

for molecule in Molecules['Proteins'] :
    Names.append(molecule)
    Mass.append(ProteinMass(Molecules['Proteins'][molecule]['Sequence'])[0])

for molecule in Molecules['Solutes'] :
    Names.append(molecule)
    Mass.append(Molecules['Solutes'][molecule]['Mass'])

MoleculesTable.index = Names
MoleculesTable['Mass (g/mol)'] = Mass
MoleculesTable['Mass (g/mol)'] = MoleculesTable['Mass (g/mol)'].round(decimals=1)

MoleculesTable = MoleculesTable