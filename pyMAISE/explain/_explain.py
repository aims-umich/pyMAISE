import numpy as np
import pandas as pd
from pyMAISE.explain.shap.explainers import KernelExplainer
from pyMAISE.explain.shap.explainers import GradientExplainer
from pyMAISE.explain.shap.explainers import DeepExplainer
from pyMAISE.explain.shap.plots._beeswarm import summary_legacy as summary_plot
import matplotlib.pyplot as plt


# mir: New Changes to make bar plotter outside the class and more agnostic
def plot_bar_with_labels(df, fig=None, ax=None):
    """
    Creates a bar plot where the first column of the DataFrame is used as the bar values,
    and the second column determines if '+++' or '---' is printed on top of the columns.
    
    Parameters:
    df (pd.DataFrame): A DataFrame with at least two columns. 
                       The first column is used for the bar height,
                       the second column determines the label.
    """
    # Create the bar plot
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    bars = ax.bar(df.index, df.iloc[:, 0], capsize=4, width=0.3, color='darkorchid')
    
    # Add +++ or --- on top of each bar based on the second column's values
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if df.iloc[i, 1] > 0:
            label = '+++'
        else:
            label = '---'
        ax.text(bar.get_x() + bar.get_width() / 2, height, label, 
                ha='center', va='bottom', fontsize=12)
    
    # Labeling and displaying the plot
    ax.set_ylabel('Mean of |SHAP Values| and Net +/- Effect')
    ax.set_title('Absolute Mean Importance', loc='center')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()


#-------------------------------------
# mir: the class, started from your version
#-------------------------------------

class ShapExplainers:
    def __init__(self, 
                 base_model, 
                 X,  
                 feature_names=None,
                 output_names=None,
                 seed=None, 
                 **model_params):
        """
        A SHAP Explainer class to evaluate ad-hoc model explainability with three
        methods: Kernel SHAP, DeepLIFT, and Integrated Gradients

        Parameters:
        - base_model: A model with method .predict. Accepted models are TF/Keras NN
        - X: Input dataset used to estimate SHAP values, it may include
             training inputs (Xtrain), test inputs (Xtest), or a combination of both.
        - seed: Fixing the seed when sampling data from test set array
        - model_params: Additional parameters for the base model.
        """
        self.model = base_model
        self.X = X
        self.shap_raw = {}
        self.shap_samples = {}
        self.feature_names = feature_names
        self.output_names = output_names
        if seed:
            np.random.seed(seed)
        
        self.n_features= self.X.shape[1]
        if self.feature_names is None:
            self.feature_names = np.array(["FEATURE " + str(i) for i in range(self.n_features)])
        # else:
        #     self.feature_names = _get_idx(self) 

        #infer number of outputs from the model prediction of two samples
        self.n_outputs=self.model.predict(self.X[0:2,:]).shape[1]
        if self.output_names is not None: 
            assert len(self.output_names) == self.n_outputs   
        else:
            self.output_names = np.array(["OUTPUT " + str(i) for i in range(self.n_outputs)])

    ## ADD SHAP METHODS HERE!
    def DeepLIFT(self, nsamples=None):
        """
        Fit a DeepLIFT explainer to evaluate SHAP coeffiicents (only for neural networks)

        Parameters:
        - nsamples: Number of samples used to estimate the DeepLIFT importances  
                    if it is different than using all samples in X
        """
        # Come up with randomized indices to grab from the Xtest array
        if nsamples is not None:
            test_indices = np.random.choice(self.X.shape[0], size=nsamples, replace=False)
            # Grab the sample points
            test_x_sample = self.X[test_indices]
        else:
            test_x_sample = self.X.copy()
            
        # Get the shap values for DeepLift using the sample Xtest set
        self.deep_lift = DeepExplainer(self.model, data=[self.X])
        deepshap_values = self.deep_lift.shap_values(test_x_sample)
        # Add those values to the dictionary of shap explainer results
        self.shap_raw["DeepLIFT"] = deepshap_values
        # Add the samples to the dictionary of shap explainer samples
        self.shap_samples["DeepLIFT"] = test_x_sample

    def IntGradients(self, nsamples=None):
        """
        Fit a Integrated Gradient explainer to evaluate SHAP coeffiicents

        Parameters:
        - nsamples: Number of test samples used to estimate the IG importances
                    if it is different than using all samples in X
        """
        if nsamples is not None:
            test_indices = np.random.choice(self.X.shape[0], size=nsamples, replace=False)
            test_x_sample = self.X[test_indices]
        else:
            test_x_sample = self.X.copy()
            
        self.ig = GradientExplainer(self.model, data=[self.X])
        igshap_values = self.ig.shap_values(test_x_sample)
        
        self.shap_raw["IG"] = igshap_values
        self.shap_samples["IG"] = test_x_sample
        
    def KernelSHAP(self, n_background_samples=500,
                           n_test_samples=200,
                           n_bootstrap=200):
        """
        Fit a Kernel SHAP explainer to evaluate SHAP coeffiicents

        Parameters:
        - n_background_samples: Number of training samples used as background 
                                for integrating out features
        - n_test_samples: Number of test samples used to estimate the Kernel SHAP importances
        - n_bootstrap: Number of times to re-evaluate the model when explaining 
                      each prediction. More samples lead to lower variance 
                      estimates of the SHAP values. 
        """
        #mir: To-do later, we will find a better way to determine the background
        #samples in the future, for now, the background samples should be 
        #limited to 1000. 
        indices = np.random.choice(self.X.shape[0], 
                                   size=n_background_samples + n_test_samples, 
                                   replace=False)        
        background_data = self.X[indices[0:n_background_samples]]
        test_data = self.X[indices[n_background_samples:]]
        
        self.kernel_e = KernelExplainer(self.model.predict, data=background_data)
        kernel_shap_values = self.kernel_e.shap_values(test_data, nsamples=n_bootstrap)
                
        self.shap_raw["KernelSHAP"] = kernel_shap_values    
        self.shap_samples["KernelSHAP"] = test_data

    def postprocess_results(self): 
        self.shap_mean = {}
        self.shap_net_effect = {}
        for i, (key, value) in enumerate(self.shap_raw.items()):
            self.shap_mean[key] = pd.DataFrame(np.abs(self.shap_raw[key]).mean(axis=0),
                                               columns=self.output_names,
                                               index=self.feature_names)
            
            #total_effect is now a 2D array of features x outputs
            total_effect = self.shap_raw[key].sum(axis=0) 
            #normalize the values while preserving the sign
            norm_effect = total_effect / np.sum(np.abs(total_effect), axis=0)
            self.shap_net_effect[key] = pd.DataFrame(norm_effect,
                                               columns=self.output_names,
                                               index=self.feature_names)
    
    #mir: I realized how complex this function became as now if you want to make 
    # a simple change in the plotting, we need to do it four times, it is fine now to keep
    #like this to get the results for the paper, but then lets talk how to make 
    #it more flexible and simplified
    def plot(self, output_name=None, output_index=None, method=None, max_display=20):
        """
        Makes a beeswarm plot and bar plot for each shap method, or to make one for a particular method. If no output_index is given, make a plot for each. 

        Parameters:
        - output_name: The name of the output variable the user is interested in plotting. Must be defined in output_names.
        - output_index: The index of the output variable the user is interested in plotting.
        - method: The key of the shap_raw array for the shap method a user wishes to plot. Options include: "DeepLIFT", "KernelSHAP", or "IG"
        - max_display: The maximum number of input features that will be displayed on a beeswarm plot.
        """
        # Check to see if post_process results has been run before making plots
        if self.shap_mean is None or self.shap_net_effect is None:
            emsg = "Results have not been post-processed. Please run post_process() method on your explain object prior to attempting plotting."
            raise AttributeError(emsg)
        
        # Get output index from output name 
        if output_index is None and output_name is not None:
            names = np.array(self.output_names)
            output_index = np.argwhere(names==output_name)
        
        # Check to make sure output name (if given) actually exists
        if output_name is not None and output_name not in self.output_names:
            emsg = f"The output you requested is not defined for this model. Valid output names include: {self.output_names}."
            raise NameError(emsg)
        
        figsize = (18, 8)
        
        # If there's no method listed and no output index listed, make a plot for all methods and all outputs
        if method==None and output_index==None:
            # Loop over all outputs
            for i in range(self.n_outputs):
                # Loop over all methods
                for key, val in self.shap_raw.items():
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

                    # Make the beeswarm plot
                    fig.sca(ax1)
                    summary_plot(val[:,:,i], 
                                features=self.shap_samples[key], 
                                feature_names=self.feature_names,
                                show=False, 
                                plot_size=None,
                                max_display=max_display)
                    ax1.set_title('Beeswarm Summary', loc='center')

                    # Make the bar plot
                    df_mean_sorted= self.shap_mean[key].iloc[:,i].sort_values(ascending=False)
                    df_neteffect_sorted = self.shap_net_effect[key].iloc[:,0].loc[df_mean_sorted.index]
                    df_combined = pd.concat([df_mean_sorted, df_neteffect_sorted], axis=1)
                    fig.sca(ax2)
                    plot_bar_with_labels(df_combined, fig=fig, ax=ax2)
                    
                    # Save and return the figure
                    fig.suptitle(f'{key} {self.output_names[i]}', 
                                 fontsize='x-large', fontweight='bold', y=1.02)
                    fig.tight_layout()
                    
                    #mir: only make this a flag to save, sometimes people get annoyed when plots 
                    #generated by default in their directory
                    fig.savefig(f'{key}_output_{i}.png', dpi=300)
                     
        # Looking for a single output for all methods
        elif method==None:
            for key, val in self.shap_raw.items():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
                # Make the beeswarm plot
                fig.sca(ax1)
                summary_plot(val[:,:,output_index], 
                            features=self.shap_samples[key], 
                            feature_names=self.feature_names,
                            show=False, 
                            plot_size=None,
                            max_display=max_display)
                ax1.set_title('Beeswarm Summary')

                # Make the bar plot
                df_mean_sorted= self.shap_mean[key].iloc[:,output_index].sort_values(ascending=False)
                df_neteffect_sorted = self.shap_net_effect[key].iloc[:,0].loc[df_mean_sorted.index]
                df_combined = pd.concat([df_mean_sorted, df_neteffect_sorted], axis=1)
                fig.sca(ax2)
                plot_bar_with_labels(df_combined, fig=fig, ax=ax2)
                fig.suptitle(f'{key} {self.output_names[output_index]}', 
                             fontsize='x-large', fontweight='bold', y=1.02)
                fig.savefig(f'{key}_output_{output_index}.png', dpi=300)
            
        # Looking for a single method, all outputs
        elif output_index==None:
            for i in range(self.n_outputs):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
                fig.sca(ax1)
                summary_plot(self.shap_raw[method][:,:,i], 
                        features=self.shap_samples[method], 
                        feature_names=self.feature_names,
                        show=False, 
                        plot_size=None,
                        max_display=max_display)
                ax1.set_title('Beeswarm Summary')
                
                # Make the bar plot
                df_mean_sorted= self.shap_mean[method].iloc[:,i].sort_values(ascending=False)
                df_neteffect_sorted = self.shap_net_effect[method].iloc[:,0].loc[df_mean_sorted.index]
                df_combined = pd.concat([df_mean_sorted, df_neteffect_sorted], axis=1)
                fig.sca(ax2)
                plot_bar_with_labels(df_combined, fig=fig, ax=ax2)                
                fig.suptitle(f'{method} {self.output_names[i]}',
                             fontsize='x-large', fontweight='bold', y=1.02)
                fig.savefig(f'{method}_output_{i}.png')
            
        # Looking for a specific output from a specific method
        else:
            #mir: I did not test this block as I was not sure how to invoke it, please double check
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            fig.sca(ax1)
            summary_plot(self.shap_raw[method][:,:,output_index], 
                        features=self.shap_samples[method], 
                        feature_names=self.feature_names,
                        show=False, 
                        plot_size=None,
                        max_display=max_display)
            ax1.set_title('Beeswarm Summary')
            # Make the bar plot
            df_mean_sorted= self.shap_mean[method].iloc[:,output_index].sort_values(ascending=False)
            df_neteffect_sorted = self.shap_net_effect[method].iloc[:,0].loc[df_mean_sorted.index]
            df_combined = pd.concat([df_mean_sorted, df_neteffect_sorted], axis=1)
            fig.sca(ax2)
            plot_bar_with_labels(df_combined, fig=fig, ax=ax2)   
            fig.suptitle(f'{method} {self.output_names[output_index]}', 
                         fontsize='x-large', fontweight='bold', y=1.02)
            fig.savefig(f'{method}_output_{output_index}.png', dpi=300)
            return fig, ax1, ax2
            

