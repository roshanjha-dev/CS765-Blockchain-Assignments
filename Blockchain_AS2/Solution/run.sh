
#!/bin/bash

# Define fixed values
z0=5
Ttx=20


# Define mappings for z1 -> (x, y)
declare -A x_y_map=(
    [100]="10 10"
    [90]="15 15"
    [80]="20 20"
    [70]="30 30"
    [60]="40 40"
    [50]="75 75"
    [40]="100 100"
    [30]="150 150"
    [20]="150 150"
    [10]="200 200"
    [0]="150 150"
)

# Loop over n from 10 to 100, increasing by 10
for ((Tt=30; Tt <=100; Tt+=30));do
	for ((n=10; n<=100; n+=10)); do
		# Create directory for the current n
		dir_name="timeout_${Tt}/${n}_peers"
		mkdir -p "$dir_name"

		# Loop over z1 from 0 to 100, increasing by 10
		for ((z1=100; z1>=0; z1-=10)); do
		    # Get x and y values from the mapping
		    x_y=(${x_y_map[$z1]})
		    x=${x_y[0]}
		    y=${x_y[1]}

		    # Define output image filename
		    img_file="visual_${n}_$((100 - z1)).png"
            out_file="output_${n}_$((100 - z1)).txt"

		    # Run the Python script
		    python3 our_simu.py "$n" "$z0" "$z1" "$Ttx" "$x" "$y" "$img_file" "$Tt" > "$out_file"

		    # Move the generated image to the corresponding directory
		    mv "$img_file" "$dir_name/"
            mv "$out_file" "$dir_name/"
		done
	done
done
