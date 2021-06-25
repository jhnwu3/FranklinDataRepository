# Set the output to a png file
set terminal png size 800,800
# The file we'll write to
set output 'linearRates.png'
# The graphic title
set title 'Nonlinear Rates'

# labels
set xlabel "Iteration"
set ylabel "Rate"
set grid

#plot the graphic
plot "Jay_Global_Best.txt" using 1:2 title "k1", "Bill_Global_Best.txt" using 1:3 title "k2", "Bill_Global_Best.txt" using 1:4 title "k3", "Bill_Global_Best.txt" using 1:5 title "k4", "Bill_Global_Best.txt" using 1:4 title "k3", "Bill_Global_Best.txt" using 1:5 title "k4", "Bill_Global_Best.txt" using 1:6 title "k5", "Bill_Global_Best.txt" using 1:7 title "k6"

