metadata_wgs
donor_wgs=metadata_wgs['Cell.lineID']
# donor_wgs = donor_wgs[(metadata_wgs['Tissue'] == 'iPSC') & (metadata_wgs['WGS'] == 'avail')]
donor_wgs = donor_wgs[(metadata_wgs['Tissue'] == 'iPSC') & (metadata_wgs['scRNAseq'] == 'avail')]

donor_rna = df_eff.donor_id.map(lambda s: s.split('-')[1].split('_')[0])
donor_rna = df_eff.donor_id.map(lambda s: s.split('-')[1])

pd.cut(df_eff[np.isin(donor_rna,donor_wgs)].diff_efficiency,5).value_counts()
