# Set the output to a png file
set terminal png size 800,800
# The file we'll write to
set output 'costs_dist_rand.png'
# The graphic title
set title 'Costs Based on HyperCube k'

# labels
set xlabel "Costs"
set ylabel "Count"
set grid

binwidth=5
bin(x,width)=width*floor(x/width) + width/2.0
set boxwidth binwidth

plot 'Protein_Cost_Dist_Rand.txt' using (bin($1,binwidth)):(1.0) smooth freq with boxes