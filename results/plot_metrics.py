import os
import shutil
import pandas as pd
from plotnine import *
from pathlib import Path


file_paths_20ng = ['./mallet-20ng/', './dvae-20ng/', './etm-20ng/']
file_paths_wiki = ['./mallet-wiki/', './dvae-wiki/', './etm-wiki/']


def topic_files(file_path):
    dir_names = [os.path.normpath(path).split(os.sep) for path in file_path]
    dir_names = [name for dir_name in dir_names for name in dir_name]
    for i in range(len(file_path)):
        shutil.copyfile(Path(file_path[i], "npmis.txt"), './readable-format/npmis_' + dir_names[i] + '.txt')

        reader = open(file_path[i] + 'topics.txt')
        # create list of list
        list_topics = [line.replace("\n", "").split() for line in reader.readlines()]
        list_topics = [topic[:10] for topic in list_topics]

        with open('./readable-format/topics_' + dir_names[i] + '.txt', 'w') as f:
            for j in range(len(list_topics)):
                f.write(str(list_topics[j]) + '\n')
#topic_files(file_paths_20ng)
topic_files(file_paths_wiki)


def create_boxplot(file_paths, data):
    dfs = [pd.read_csv(path + 'npmis.txt', header=None) for path in file_paths]
    models = ["G-LDA", "D-VAE", "ETM"]

    for ind in range(len(dfs)):
        for i in dfs[ind].index:
            dfs[ind].loc[i, 'model'] = models[ind]
        dfs[ind].columns = ["npmi_scores", "model"]
        dfs[ind] = dfs[ind][['model', "npmi_scores"]]
    df = pd.concat(dfs)

    ggsave(
        ggplot(df)
        + geom_boxplot(aes(x="model", y="npmi_scores", fill="model"), show_legend=False)
        + xlab("Automated")
        + ylab(data)
        + theme(
            axis_line=element_line(size=1, colour="black"),
            panel_grid_major=element_line(colour="#d3d3d3"),
            panel_grid_minor=element_blank(),
            panel_border=element_blank(),
            panel_background=element_blank(),
            plot_title=element_text(size=12, family="Tahoma", face="bold"),
            text=element_text(family="Tahoma", size=10),
            axis_text_x=element_text(colour="black", size=10),
            axis_text_y=element_text(colour="black", size=8),
            subplots_adjust={'wspace': 0.3, 'hspace': 0.5},
            strip_margin_x=0.3,
        )
        + scale_x_discrete(limits=("G-LDA", "D-VAE", "ETM"))
        + theme(figure_size=(8, 4))
        + scale_fill_brewer(type="qual", palette="Set2"),
        filename="./readable-format/model_comparison_boxplot_"+data+".pdf",
        dpi=320)

create_boxplot(file_paths_20ng, "20_Newsgroups")
create_boxplot(file_paths_wiki, "Wikipedia")
