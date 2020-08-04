## here we add the coordinates to its image 

## then we re- order them to radius and non_radius file 

## with the python file that uses regex 

#while read -rd $'\0' f; do    d="${f%/*}"; p="${d/\//_}";   mv -- "$f" "${d}/${p}_${f##*/}"; done < <(find -type f -name '*.png' -printf '%P\0')

#find . -name '*.png' -exec sh -c 'mv "$0" "$(basename $(dirname $0))-${0%.PNG}$"' {} -;


python coordinates_ordering.py --path /home/jonathan/Videos/Patient_Images_Original/adjusted_11_patients/ --moveto /home/jonathan/Videos/Patient_Images_Original/adjusted_11_patients/val/radius/
