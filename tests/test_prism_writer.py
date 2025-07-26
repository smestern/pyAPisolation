from pyAPisolation.dev.prism_writer import PrismFile
import numpy as np
import pandas as pd
import os
def test_prism_writer():
    np.random.seed(42)
    file = PrismFile()

   
    #make some random data to test with
    x = np.random.randint(0, 9, size=(100,2))
    labels = np.random.choice(['Ycol_a', 'Ycol_b'], 100)
    #turn it into a dataframe
    df = pd.DataFrame(x, columns=['rnd1', 'rnd2'])
    df['labels'] = labels
    #too double check the groupby, add 100 to all Ycol_a values in rnd1
    df.loc[df['labels'] == 'Ycol_a', 'rnd1'] += 100
    #make all Ycol_b values in rnd2 negative
    df.loc[df['labels'] == 'Ycol_b', 'rnd2'] *= -1
    #pass to make
    out = file.make_group_table('1way_group', df, groupby='labels')

    #try a 2way group
    x = np.random.rand(100, 2)
    labels = np.random.choice(['Ycol_a', 'Ycol_b'], 100)
    #turn it into a dataframe
    df = pd.DataFrame(x, columns=['rnd1', 'rnd2'])
    df['labels'] = labels
    labels2 = np.random.choice(['sub1', 'sub2'], 100)
    df['labels2'] = labels2
    #multiply the rnd1 values by 100 if they are in sub1
    df.loc[df['labels2'] == 'sub1', 'rnd1'] *= 100
    #multiply the rnd2 values by -1 if they are in sub2
    df.loc[df['labels2'] == 'sub2', 'rnd2'] *= -1
    #pass to make
    out = file.make_group_table('2way_group', df, groupby='labels', subgroupby='labels2')
    out = file.make_group_table('2way_group2', df, groupby='labels', rowgroupby='labels2')

    #try a 3way group
    x = np.random.rand(50, 2)
    labels = np.random.choice(['Ycol_a', 'Ycol_b'], 50)
    #turn it into a dataframe
    df = pd.DataFrame(x, columns=['rnd1', 'rnd2'])
    df['labels'] = labels
    labels2 = np.random.choice(['sub1', 'sub2'], 50)
    df['labels2'] = labels2
    labels3 = np.random.choice(['row1', 'row2'], 50)
    df['labels3'] = labels3
    #pass to make
    out = file.make_group_table('3way_group', df, groupby='labels', subgroupby='labels2', rowgroupby='labels3')

    #try just rowgroupby
    x = np.random.rand(50, 2)
    labels = np.random.choice(['Ycol_a', 'Ycol_b'], 50)
    #turn it into a dataframe
    df = pd.DataFrame(x, columns=['rnd1', 'rnd2'])
    df['labels'] = labels
    labels3 = np.random.choice(['row1', 'row2'], 50)
    df['labels3'] = labels3
    #pass to make
    out = file.make_group_table('rowcols', df, groupby='labels', rowgroupcols=['rnd1', 'rnd2'])

    #load the xlsx
    test_df = pd.read_csv('test.csv')
    rowgroupcols = ['Sweep 001 spike count', 'Sweep 002 spike count', 'Sweep 003 spike count', 'Sweep 004 spike count', 'Sweep 005 spike count', 'Sweep 006 spike count', 'Sweep 007 spike count', 'Sweep 008 spike count', 
                    'Sweep 009 spike count', 'Sweep 010 spike count', 'Sweep 011 spike count', 'Sweep 012 spike count', 'Sweep 013 spike count', 'Sweep 014 spike count', 'Sweep 015 spike count']
    #make a group table
    out = file.make_group_table('test_group', test_df, groupby='Burst Cadex', rowgroupcols=rowgroupcols)

    #TRY A 3WAY GROUP WITH ROWGROUPCOLS
    out = file.make_group_table('3way_rowcols', test_df, cols='Sag Ratio', groupby='Burst Cadex', rowgroupby='1Afoldername', subgroupby='1Afilename')

    # #try to write it
    file.write('test.pzfx')
    print('done')
    #unfortunately, the full output has to be checked manually, but the file should be written to the current directory
    #and can be opened in PRISM
    os.path.exists('test.pzfx') #True

if __name__ == '__main__':
    test_prism_writer()