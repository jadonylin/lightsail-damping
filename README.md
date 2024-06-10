Code and data for the journal paper.

The `effderivs.py`, `lorentz.py` and `FOM.py` files are modules containing the RCWA reflection, Doppler factor and figure of merit functions, respectively.

The `optimisation.py` file runs the optimisation for a specified number of max function evaluations and CPU cores and saves the result in `optimisation.pkl`. 
For convenience, the maxima produced for each core is printed as a text file. Most of the maxima in the optimisation come from very sharp Fano resonances dragging up the 
figure of merit, so we curated the maxima until we found one with a broad bandwidth. This maxima was then run as the starting point for a local optimisation using the 
Method of Moving Asymptotes. The final maxima is shown in the figure of merit curve `opt_grating.pdf`, the plot for which was produced using code from `final_plots.ipynb`.

The dynamics data is also generated in `final_plots.ipynb` and saved in the `final_data` folder. 

