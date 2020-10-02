import seaborn
import matplotlib.pyplot as plt

from pandas import read_csv

input_path = "../input/SPD_Reports.csv";

reports = read_csv(input_path);

print("Processing ALL");
# This has to be hex to prevent the kernel from timing out, kde is just too slow with that much data.
all_plot = seaborn.jointplot(x="Longitude", y="Latitude", data=reports, kind="hex", size=8);

for offense_type in reports["Offense Type"].unique():
    print("Processing " + offense_type);
    data = reports[reports["Offense Type"] == offense_type];
    if len(data) > 2:
        plot = seaborn.jointplot("Longitude", "Latitude", data=data, kind="kde", size=8, xlim=all_plot.ax_joint.get_xlim(), ylim=all_plot.ax_joint.get_ylim());
        plt.subplots_adjust(top=0.93);
        plot.fig.suptitle(offense_type);
        plot.savefig(offense_type.replace("/", "-").replace(" ", "").replace(",", "").replace("[", "").replace("]", "")  + ".png");
        plot.fig.clf();
        plt.close();