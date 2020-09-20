import os
import time
import sys
# conda install -c conda-forge progressbar
import progressbar
# Download this from http://pypi.python.org/pypi/futures
#from concurrent import futures
# rdkit cheminformania
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import TorsionFingerprints
from rdkit.ML.Cluster import Butina



#mol = Chem.MolFromSmiles('CCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCC)OC(=O)CCCCCCCC')
#molH = Chem.AddHs(mol)
#confIds = AllChem.EmbedMultipleConfs(molH, num_pot_confs, pruneRmsThresh = 0.5)
#molH.GetNumConformers()

def gen_conformers(mol, numConfs=100, maxAttempts=1000, pruneRmsThresh=0.1, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, enforceChirality=True):
    """Improving Conformer Generation for Small Rings and Macrocycles
    Based on Distance Geometry and Experimental Torsional-Angle
    Preferences. DOI: 10.1021/acs.jcim.0c00025"""
    # ETDG, ETKDG, ETKDGv2, ETKDGv3
    cids = AllChem.EmbedMultipleConfs(mol,
                                    numConfs=100,
                                    params=AllChem.ETKDGv3())
#    cids = AllChem.EmbedMultipleConfs(mol,
#                                    clearConfs=True,
#                                    numConfs=100,
#                                    pruneRmsThresh=1)
    print (len(cids));
    print(mol.GetNumConformers())
    return list(cids)

"""	ids = AllChem.EmbedMultipleConfs(mol,numConfs=numConfs,axAttempts=maxAttempts,
                                    pruneRmsThresh=pruneRmsThresh,
                                    useExpTorsionAnglePrefs=useExpTorsionAnglePrefs,
                                    useBasicKnowledge=useBasicKnowledge,
                                    enforceChirality=enforceChirality,
                                    numThreads=0)"""



def write_conformers_to_sdf(mol, filename, rmsClusters, conformerPropsDict, minEnergy):
	w = Chem.SDWriter(filename)
	for cluster in rmsClusters:
		for confId in cluster:
			for name in mol.GetPropNames():
				mol.ClearProp(name)
			conformerProps = conformerPropsDict[confId]
			mol.SetIntProp("conformer_id", confId + 1)
			for key in conformerProps.keys():
				mol.SetProp(key, str(conformerProps[key]))
			e = conformerProps["energy_abs"]
			if e:
				mol.SetDoubleProp("energy_delta", e - minEnergy)
			w.write(mol, confId=confId)
	w.flush()
	w.close()

def calc_energy(mol, conformerId, minimizeIts):
	ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conformerId)
	ff.Initialize()
	ff.CalcEnergy()
	results = {}
	if minimizeIts > 0:
		results["converged"] = ff.Minimize(maxIts=minimizeIts)
	results["energy_abs"] = ff.CalcEnergy()
	return results

def cluster_conformers(mol, mode="RMSD", threshold=2.0):
	if mode == "TFD":
		dmat = TorsionFingerprints.GetTFDMatrix(mol)
	else:
		dmat = AllChem.GetConformerRMSMatrix(mol, prealigned=False)
	rms_clusters = Butina.ClusterData(dmat, mol.GetNumConformers(), threshold, isDistData=True, reordering=True)
	return rms_clusters

def align_conformers(mol, clust_ids):
	rmslist = []
	AllChem.AlignMolConformers(mol, confIds=clust_ids, RMSlist=rmslist)
	return rmslist


if len(sys.argv) < 3:
    print("\nUsage:\n\tconf_gen.py [file.sdf] [num conformers] \n")
    exit()

input_file = sys.argv[1]
numConfs = int(sys.argv[2])
maxAttempts = 1000
pruneRmsThresh = 0.1
# cluster method: (RMSD|TFD) = RMSD
clusterMethod = "RMSD"
clusterThreshold = 2.0
minimizeIterations = 0


#mol = Chem.MolFromSmiles('CC[C@H](C)[C@@H]1NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@H](CC(=O)O)NC(=O)[C@@H]2CCCN2C(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](C(C)C)NC(=O)[C@H](Cc2c[nH]cn2)NC1=O')
#cids = AllChem.EmbedMultipleConfs(mol, numConfs=50, maxAttempts=1000, pruneRmsThresh=0.1)
#print(len(cids))

suppl = Chem.ForwardSDMolSupplier(input_file)
i=0
for mol in suppl:
    i = i+1

    mol = Chem.MolFromSmiles('CC[C@H](C)[C@@H]1NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@H](CC(=O)O)NC(=O)[C@@H]2CCCN2C(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](C(C)C)NC(=O)[C@H](Cc2c[nH]cn2)NC1=O')

    print ("--->",mol)
    if mol is None: continue
    m = Chem.AddHs(mol)
    # generate the confomers
    conformerIds = gen_conformers(m, numConfs, maxAttempts, pruneRmsThresh, True, True, True)
    conformerPropsDict = {}
    for conformerId in conformerIds:
        # energy minimise (optional) and energy calculation
        props = calc_energy(m, conformerId, minimizeIterations)
        conformerPropsDict[conformerId] = props
    # cluster the conformers
    rmsClusters = cluster_conformers(m, clusterMethod, clusterThreshold)

    print ("Molecule", i, ": generated", len(conformerIds), "conformers and", len(rmsClusters), "clusters")
    rmsClustersPerCluster = []
    clusterNumber = 0
    minEnergy = 9999999999999
    for cluster in rmsClusters:
        clusterNumber = clusterNumber+1
        rmsWithinCluster = align_conformers(m, cluster)
        for conformerId in cluster:
            e = props["energy_abs"]
            if e < minEnergy:
                minEnergy = e
            props = conformerPropsDict[conformerId]
            props["cluster_no"] = clusterNumber
            props["cluster_centroid"] = cluster[0] + 1
            idx = cluster.index(conformerId)
            if idx > 0:
                props["rms_to_centroid"] = rmsWithinCluster[idx-1]
            else:
                props["rms_to_centroid"] = 0.0

    write_conformers_to_sdf(m, str(i) + ".sdf", rmsClusters, conformerPropsDict, minEnergy)
