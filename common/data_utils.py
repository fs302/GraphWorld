
data_dir = "/Users/shenfan/Code/Project/GraphWorld/data/"

data_file = {"facebook": "facebook/facebook.txt",
             "arxiv": "arxiv/arxiv.txt",
             "BlogCatalog": "BlogCatalog/edges_format2.txt",
             "lyb": "lyb/lyb.e",
             "USAir": "USAir/USAir.txt"
             }


def get_data_path(dataset_name):
    return data_dir + data_file[dataset_name]


if __name__ == "__main__":
    print(get_data_path("facebook"))
