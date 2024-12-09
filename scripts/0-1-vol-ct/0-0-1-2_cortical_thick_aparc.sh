#!/bin/bash

rm ct_temp.txt
rm ct_output.txt
file_list=($(cat subj.txt)) 


touch ct_output.txt
for subj in "${file_list[@]}"
do
	volume=$(grep -v "^#" $subj | awk '{print $5}')
	echo -e "${subj:4:8}\n$volume" > ct_temp.txt
	paste ct_temp.txt ct_output.txt > ct_pasted.txt
	mv ct_pasted.txt ct_output.txt
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
