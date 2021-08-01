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
plot "<(sed -n '1,10p' Bill_Global_Best.txt)" using 1:7 title "Linear Cost" with points pointtype 5

