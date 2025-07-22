module load cuda

for FILE in profiles/*.nsys-rep; do
    filename=$(basename -- "$FILE")
    stem="${filename%.*}"
    echo profiles/$stem
    nsys stats -o profiles/$stem -r cudaapisum -r gpukernsum $FILE
done