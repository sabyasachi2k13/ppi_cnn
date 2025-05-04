#!/bin/bash
  
search_dir="/data/work/work-pssm/test"

for entry in "$search_dir"/*
do
  echo "########################"
  echo "$entry"

  loc="/data/work/pssm/" 
  nameLen=${#search_dir}
  name="${entry:nameLen:7}"
  final="$loc$name"
  #/data/work/ncbi-blast-2.12.0+/bin/psiblast -db /data/work/db/prot -query  $entry  -out_ascii_pssm $final  -num_iterations 3  -num_threads 32 
  #echo   psiblast -db /data/work/db/prot  -query $entry  -out_ascii_pssm $final  -num_iterations 3  -num_threads 16
  
  #FILE=/etc/resolv.conf
  
  if [ -f "$final" ]; then
  	echo "$final exists."
  else 
  	echo "$final does not exist."
	psiblast -db /data/work/work-pssm/db/prot  -query $entry  -out_ascii_pssm $final  -num_iterations 3  -num_threads 4
	echo "$final does not exist."
  fi
   
done

echo done All


