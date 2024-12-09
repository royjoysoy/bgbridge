#!/bin/bash

rm temp_rh_aparc_a2009s.txt
rm output_rh_aparc_a2009s.txt
file_list=($(cat rh_aparc_a2009s_path.txt)) 


touch output_rh_aparc_a2009s.txt
for subj in "${file_list[@]}"
do
	volume=$(grep -v "^#" $subj | awk '{print $4}')
	#echo -e "${subj:38:47}\n$volume" > temp_lh_aparc.txt
    echo -e "${subj#*sub-}\n$volume" | cut -d/ -f1 > temp_rh_aparc_a2009s.txt
	paste temp_rh_aparc_a2009s.txt output_rh_aparc_a2009s.txt > pasted_rh_aparc_a2009s.txt
	mv pasted_rh_aparc_a2009s.txt output_rh_aparc_a2009s.txt
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
