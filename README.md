# Virtual Lesion
This series of notebooks calculates the likeliness of the existens of a fiber bundle with LiFEs Virtual Lesion approach<sup>[1]</sup>. The probability will be defined based on the Strength of Evidence.

Steps:

<ol>
    <li>Generate fiber tracks with MRtrix3 from ROI1 to ROI2 and vise versa</li>
    <li>Remove fibers which pass through the ROIs, but do not stop in them</li>
    <li>Combine all valid fiber tracks into one streamline set</li>
    <li>Cluster the streamlines with dipys QuickBundles algorithm to remove outliers</li>
    <li>Calculate the neighborhood of the candidate streamlines (Voxels where the streamlines pass through)</li>
    <li>Find fibers in whole brain fiber track that pass through the neibghorhood</li>
    <li>Combine whole brain connectome with candidate streamlines into one steamline set</li>
    <li>Run the LiFE Optimization on this combined streamline set and retrieve RMSE between the original data set and the LiFE prediction</li>
    <li>Run the LiFE Optimization and RMSE calculation again on the whole brain, but this time without the candidate streamlines</li>
    <li>Calculating the normalized difference of the two RMSEs provides the Strength of Evidence</li>
    <li>Save the LiFE optimized candidate streamlines between the two ROIs</li>
</ol>

**<small>Do not run the parallization of all cores. This notebook is parallalized by subjects. Each process needs a lot of memory.</small>

<sup>[1]</sup> <i>Pestilli et al. [PMID: 25194848] and Leong et al. [PMID: 26748088]</i>
