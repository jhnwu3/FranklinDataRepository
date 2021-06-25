# Set the output to a png file
set terminal png size 800,800
# The file we'll write to
set output 'linearCosts.png'
# The graphic title
set title 'Linear Costs'

# labels
set xlabel "Iteration"
set ylabel "Cost"
set grid

#plot the graphic
awk "NR>=0 && NR<=15" Bill_Global_Best.txt > processedLin.txt
plot "processedLin.txt" using 1:7 title "Linear Cost"

