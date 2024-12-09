#!/bin/bash

rm temp_lh_aparc.txt
rm output_lh_aparc.txt
file_list=($(cat lh_aparc_path.txt)) 


touch output_lh_aparc.txt
for subj in "${file_list[@]}"
do
	volume=$(grep -v "^#" $subj | awk '{print $4}')
	#echo -e "${subj:38:47}\n$volume" > temp_lh_aparc.txt
    echo -e "${subj#*sub-}\n$volume" | cut -d/ -f1 > temp_lh_aparc.txt
	paste temp_lh_aparc.txt output_lh_aparc.txt > pasted_lh_aparc.txt
	mv pasted_lh_aparc.txt output_lh_aparc.txt
 	#echo -e "${subj:5:7}\n$volume" > temp.csv
	
done




#with open('aseg.stats','r') as stats_file:
#	lines = stats_file.readlines()
#data = lines[80:]
#print(data)
#with open('aseg.csv','w') as csv_file:
#		stats_reader = csv.reader(data, delimiter =',')
#		csv_writer = csv.writer(csv_file, delimiter=',')
#
#		for row in stats_reader:
#			csv_writer.writerow(row)

#df = pd.read_csv('aseg.csv', delimiter = ',')
#print(df.iloc[:,0])

#print(df.shape)
