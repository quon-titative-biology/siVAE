import gseapy as gp
import urllib
import requests

def load_geneset(name: str,
                 link: str = None,
                 download: bool = True):
    """
    Input:
        name: str, if downloading using defaut links ('scsig','Hallmark','KEGG')
        link: str, either url for downloading gmt or dir to gmt
        download: boolean
    Output:
        Dictionary mapping geneset name to genes
    """
    if download:
        name2link={}
        #
        name2link['scsig']    = 'https://data.broadinstitute.org/gsea-msigdb/msigdb/supplemental/scsig/1.0.1/scsig.all.v1.0.1.symbols.gmt'
        name2link['Hallmark'] = 'https://data.broadinstitute.org/gsea-msigdb/msigdb/release/7.5.1/h.all.v7.5.1.symbols.gmt'
        name2link['KEGG']     = 'https://data.broadinstitute.org/gsea-msigdb/msigdb/release/7.5.1/c2.cp.kegg.v7.5.1.symbols.gmt'
        name2link['Biocarta'] = 'https://data.broadinstitute.org/gsea-msigdb/msigdb/release/7.5.1/c2.cp.biocarta.v7.5.1.symbols.gmt'
        #
        filename = f'{name}.gmt'
        link = name2link[name] if link is None else link
        _ = urllib.request.urlretrieve(link,filename)
        #
    else:
        filename = link
        #
    gs_dict = gp.parser.gsea_gmt_parser(filename)
    #
    return gs_dict


# def
#             pre_res = gp.prerank(rnk=df, gene_sets=gs_dir, outdir=outdir)
#             df_pre = pre_res.res2d.sort_values('fdr')
#             df_plot['Geneset'] = 'None'
#             for gs_,genes in zip(df_pre.index[:5][::-1],df_pre.genes[:5][::-1]):
#                 genes_list = genes.split(';')
#                 df_plot['Geneset'].loc[np.isin(df_plot.index,genes_list)] = gs_

# gs_name_mapping = {k: "_".join(k.split('_')[3:-1]) for k in gs_dict.keys()}
# gs_dict_new = {gs:np.array([]) for gs in np.unique([v for v in gs_name_mapping.values()])}
# for gs_name,gs_name2 in gs_name_mapping.items():
#     gs_dict_new[gs_name2] = np.union1d(gs_dict_new[gs_name2],gs_dict[gs_name])

if __name__ == '__main__':
    pass
