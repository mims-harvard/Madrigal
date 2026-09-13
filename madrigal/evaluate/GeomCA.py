import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.cm as cm

from timeit import default_timer as timer
from datetime import timedelta
import gudhi
from scipy.spatial.distance import cdist  
import logging
import pickle
import os
import umap

# -------------------------------------------------------------------------- #
# Matplotlib settings
# -------------------------------------------------------------------------- # 
SMALL_SIZE = 12
MEDIUM_SIZE = 15

plt.rc('font', size=SMALL_SIZE)           # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)      # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)    # fontsize of the figure title
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

class GeomCA():
    def __init__(self, R, E, param_dict, load_existing=False, **kwargs): 
        self.seed = param_dict['random_seed']
        np.random.seed(self.seed)

        # Network parameters
        self.Rdist_percentile = param_dict['Rdist_percentile']
        self.gamma = param_dict['gamma']
        self.comp_consistency_threshold = param_dict['comp_consistency_threshold'] 
        self.comp_quality_threshold = param_dict['comp_quality_threshold'] 
        
        # Visualisation
        self.RE_2D_projection = None
        self.R_color, self.E_color = 'C0', 'C1'
        
        # Paths
        self.path = param_dict['experiment_path'] if param_dict['experiment_path'] is not None else './temp/'
        if kwargs.get('subfolder'):
            self.path = os.path.join(self.path, kwargs.get('subfolder'))
        self.save_name = param_dict['experiment_filename_prefix']
        
        if load_existing:
            self.init_folders()
            self.init_loggers()
            self.load_GeomCA()
        
        else:                     
            self.init_folders()
            self.init_loggers()
            
            # Use a predetermined epsilon or estimate it
            if 'epsilon' in param_dict.keys():
                self.epsilon = param_dict['epsilon']
                print('Epsilon set to: {}'.format(self.epsilon))
            else:
                self.epsilon = self.estimate_distance(R, param_dict['Rdist_ratio'])
                print('Epsilon estimated to: {}'.format(self.epsilon))
            
            # Reduce points and build epsilon graph
            self.delta = self.epsilon * self.gamma
            self.plot_ptns(R, E, filename='originals')
            self.sparsify_points(R, param_dict['reduceR'], E, param_dict['reduceE'], 
                                 param_dict['log_reduced'], param_dict['sparsify'], 
                                 param_dict['n_Rsamples'], param_dict['n_Esamples'])
            self.RE_network = self.build_network()
            
            # Log the parameters
            self.network_params_logger = {
                'num_R_points': self.R_pts_idx,
                'num_E_points': self.E_pts_idx,
                'num_RE_points': self.R_pts_idx + self.E_pts_idx,
                'epsilon': self.epsilon, 
                'gamma': self.gamma,
                'comp_consistency_threshold': self.comp_consistency_threshold,
                'comp_quality_threshold': self.comp_quality_threshold}

    # -------------------------------------------------------------------------- #
    # Inits and logs
    # -------------------------------------------------------------------------- #
    def init_folders(self, GeomCA_subfolder=''):
        """
        Initialises experiment folders.
        :param GeomCA_subfolder: name of a subfolder to save results in.
        """
        self.ptns_path = os.path.join(self.path, 'point_clouds')
        self.results_path = os.path.join(self.path, 'GeomCA_results', GeomCA_subfolder)
        self.logs_path = os.path.join(self.path, 'GeomCA_logs')
        
        if not os.path.exists(self.ptns_path):
            os.makedirs(self.ptns_path)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
    
    def init_loggers(self):
        """Initialises eperiment loggers."""
        logging.basicConfig(
            level=logging.INFO, 
            format='[[%(asctime)s]] :: %(message)s', 
            datefmt='%m/%d/%Y %I:%M:%S %p',
            filename=os.path.join(self.path, self.save_name + 'params.log'),
            filemode='a')
        logging.getLogger('matplotlib.font_manager').disabled = True             
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.comp_stats_logger = {}
        self.network_stats_logger = {
            'num_R_points_in_qualitycomp': 0,
            'num_E_points_in_qualitycomp': 0,
            'precision': None, 
            'recall': None,
            'network_consistency': None, 
            'network_quality': None}

    def load_GeomCA(self):
        """Loads an existing GeomCA directory."""
        # Load comp_stats_logger
        logger_path = os.path.join(self.logs_path, self.save_name + 'components_stats.pkl')
        with open(logger_path, 'rb') as f:
            self.comp_stats_logger = pickle.load(f)
            print('Components statistics loaded.')
            
        # Load calculated network parameters
        params_path = os.path.join(self.logs_path, self.save_name + 'network_parameters.pkl')
        with open(params_path, 'rb') as f:
            params_dict = pickle.load(f)
            self.R_pts_idx = params_dict['num_R_points']
            self.E_pts_idx = params_dict['num_E_points']
            self.epsilon = params_dict['epsilon']
            self.gamma = params_dict['gamma']
            self.delta = self.epsilon * self.gamma
            self.comp_consistency_threshold = params_dict['comp_consistency_threshold']
            self.comp_quality_threshold = params_dict['comp_quality_threshold']
            print('Epsilon graph parameters loaded.')
            
            self.network_params_logger = {
                'num_R_points': self.R_pts_idx,
                'num_E_points': self.E_pts_idx,
                'num_RE_points': self.R_pts_idx + self.E_pts_idx,
                'epsilon': self.epsilon, 
                'gamma': self.gamma,
                'comp_consistency_threshold': self.comp_consistency_threshold,
                'comp_quality_threshold': self.comp_quality_threshold}
            
        # Load network stats logger
        params_path = os.path.join(self.logs_path, self.save_name + 'network_stats.pkl')
        with open(params_path, 'rb') as f:
            self.network_stats_logger = pickle.load(f)
            print('Network statistics loaded.')
        
        
        # Load sparseR, sparseE if they exists
        ptns_path = os.path.join(self.logs_path, self.save_name + 'reduced_points.pkl')
        if os.path.isfile(ptns_path):
            with open(ptns_path, 'rb') as f:
                ptns_dict = pickle.load(f)
                self.sparseR = ptns_dict['sparseR']
                self.sparseE = ptns_dict['sparseE']
            print('Sparse representations loaded.')
        
    def log(self, msg, var=None, sub=False, nl=False):
        """
        Writes a log message.
        :param msg: message to log.
        :param var: variable value to log.
        :param sub: if indent the log message.
        :param nl: if creating new line before log message.
        """
        log_msg = '{0}{1}{2}{3}'.format(
            '\n\n' if nl else '', 
            ' - ' if sub else '',
            msg,
            ': ' + str(var) if var != None else '')
        self.logger.info(log_msg)

    def save_geomCA_logs(self):
        """Saves GeomCA statistics."""
        self.log_components_stat()
        self.log_network_parameters()
        self.log_network_stats()
        self.log_to_txt()  

    def log_reduced_pnts(self):
        """Saves reduced R and E points."""
        path = os.path.join(self.logs_path, self.save_name + 'reduced_points.pkl')
        with open(path, 'wb') as f:
            pickle.dump({'sparseR': self.sparseR, 'sparseE': self.sparseE}, f)
    
    def log_components_stat(self):
        """Saves components statistics."""
        path = os.path.join(self.logs_path, self.save_name + 'components_stats.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self.comp_stats_logger, f)
        
    def log_network_parameters(self):
        """Saves chosen/estimated network parameters."""
        path = os.path.join(self.logs_path, self.save_name + 'network_parameters.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self.network_params_logger, f)
    
    def log_network_stats(self):
        """Saves network statistics."""
        precision = self.network_stats_logger['num_E_points_in_qualitycomp']/self.E_pts_idx
        recall = self.network_stats_logger['num_R_points_in_qualitycomp']/self.R_pts_idx
        self.network_stats_logger['precision'] = precision
        self.network_stats_logger['recall'] = recall
        network_consistency, network_quality = self.get_network_consistency_and_quality()
        self.network_stats_logger['network_consistency'] = network_consistency
        self.network_stats_logger['network_quality'] = network_quality
        
        print('Precision: {0}\nRecall: {1}\nNetwork consistency: {2}\nNetwork quality: {3}'.format(
            precision, recall, network_consistency, network_quality))
        
        path = os.path.join(self.logs_path, self.save_name + 'network_stats.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self.network_stats_logger, f)
    
    def log_to_txt(self):
        path = os.path.join(self.results_path, self.save_name + 'network_stats.txt')
        with open(path, 'w') as f:
            f.writelines(['--- GeomCA parameters\n'])
            f.writelines(['{0}: {1}\n'.format(k, v) for k, v in self.network_params_logger.items()])
            
            f.writelines(['\n--- GeomCA network stats\n'])
            f.writelines(['{0}: {1}\n'.format(k, v) for k, v in self.network_stats_logger.items()])
            
            f.writelines(['\n--- GeomCA comp stats\n'])
            f.writelines(['c(G{0}): {1:.2f}, q(G{0}): {2:.2f}, |G{0}^R|_v: {3:<4}, |G{0}^E|_v: {4:<4}, |G{0}|_v: {5:<4} \n'.format(
                comp_id, round(v['comp_consistency'], 2), round(v['comp_quality'], 2), 
                len(v['Ridx']), len(v['Eidx']), len(v['Ridx']) + len(v['Eidx'])) \
                for comp_id, v in self.comp_stats_logger.items()])
            
    # -------------------------------------------------------------------------- #    
    # GeomCA: graph building
    # -------------------------------------------------------------------------- #
    def estimate_distance(self, R, Rdist_ratio):
        """
        Estimates distances among points from the reference (true) set of representations R.
        :param R: array of reference points R to estimate epsilon on.
        :param Rdist_ratio: percentage of len(R) used to estimate distances.
        :return: estimated epsilon.
        """
        # Calulcate the distances
        self.log(self.estimate_distance.__name__, nl=True)
        fn_start = timer()
        Rs = R.shape[0]
        R = R.reshape(Rs, -1)
        num_est_pts = int(Rs * Rdist_ratio)
        R1idx = np.random.choice(Rs, num_est_pts)
        R2idx = np.random.choice(Rs, num_est_pts)
        self.log('data_dim', R.shape[-1], sub=True)
        self.log('# estimation points extracted', num_est_pts, sub=True)
        R_distances = cdist(R[R1idx], R[R2idx])
        R_distances = R_distances[R_distances > 0]
        epsilon = np.percentile(R_distances, self.Rdist_percentile)
        fn_end = timer()
        self.log('estimated_dist', epsilon, sub=True)
        self.log('Rdist_percentile', self.Rdist_percentile, sub=True)
        self.log('max_Rdist', np.max(R_distances), sub=True)
        self.log('min_Rdist', np.min(R_distances), sub=True)
        self.log('mean_Rdist', np.mean(R_distances), sub=True)
        self.log('std_Rdist', np.std(R_distances), sub=True)
        self.log('elapsed_time', timedelta(seconds=fn_end - fn_start), sub=True)
        print('Estimated epsilon: {0}'.format(str(np.round(epsilon, 4))))
        
        # Visualize the histogram
        self.plot_dist_hist(R_distances, 'Rdistances', epsilon)
        return epsilon
    
    def sparsify_points(self, R, reduceR, E, reduceE, log_reduced, sparsify, 
                        n_Rsamples=None, n_Esamples=None):
        """
        Reduces number of R and E points either by performing geometric spersification
        or by random subsampling.
        :param R: array of reference points R.
        :param reduceR: if reduce R points.
        :param E: array of reference points E.
        :param reduceE: if reduce E points.
        :param log_reduced: if save the reduced R and E points.
        :param sparsify: if reduction is geometric sparsification else subsampling.
        :param n_Rsamples: number of R points to subsample for reduction.
        :param n_Esamples: number of E points to subsample for reduction.
        """
        if reduceR:
            print('{0} R points...'.format('Sparsifying' if sparsify else 'Sampling') )
            print(' - Original R size: ', R.shape)
            self.sparseR = self.get_sparse_pnts(R, 'R') if sparsify else self.sample_sparse_ptns(R, n_Rsamples)
            
        else: 
            print('R points not reduced.')
            self.sparseR = R.reshape(len(R), -1)
        print(' - sparseR size: ', self.sparseR.shape)
        
        if reduceE:
            print('{0} E points...'.format('Sparsifying' if sparsify else 'Sampling') )
            print(' - Original E size: ', E.shape)
            self.sparseE = self.get_sparse_pnts(E, 'E') if sparsify else self.sample_sparse_ptns(E, n_Esamples)
        else:
            print('E points not reduced.')
            self.sparseE = E.reshape(len(E), -1)
        print(' - sparseE size: ', self.sparseE.shape)
        
        self.R_pts_idx = self.sparseR.shape[0]
        self.E_pts_idx = self.sparseE.shape[0]
        self.plot_ptns(self.sparseR, self.sparseE, 'reduced_points')
        if log_reduced:
            self.log_reduced_pnts()
            
    def sample_sparse_ptns(self, ptns, n_samples):
        """
        Randomly subsamples given array of points.
        :param ptns: array of points to be reduced.
        :param n_samples: number of points to subsample.
        :return: reduced array of points.
        """
        idxs = np.random.choice(np.arange(len(ptns)), n_samples)
        return ptns[idxs]
                
    def get_sparse_pnts(self, pnts, type):
        """
        Performs geometric sparsifications given by Definition 3.1.
        :param pnts: array of points to be sparsified.
        :param type: if points belong to R or E.
        :return: sparsified array of points.
        """
        self.log(self.get_sparse_pnts.__name__, nl=True)
        fn_start = timer()
        pnts = pnts.reshape(len(pnts), -1)
        sparse_pnts = np.array(gudhi.subsampling.sparsify_point_set(
            points=pnts, min_squared_dist=self.delta**2))
        fn_end = timer()
        print(' - {0} points sparsified.'.format(type))
        
        # Log
        self.log('sparse{0}.shape'.format(type), sparse_pnts.shape, sub=True)
        self.log('delta', self.delta, sub=True)
        self.log('elapsed_time', timedelta(seconds=fn_end - fn_start), sub=True)
        return sparse_pnts   

    def build_network(self):
        """Builds epsilon graph as in Definition 2.1."""
        print('Building epsilon graph...')
        edgelist = self.get_epsilon_edges()
        print(' - skeleton size: {}'.format(len(edgelist)))
        graph = nx.Graph()
        graph.add_nodes_from(np.arange(self.R_pts_idx + self.E_pts_idx))
        graph.add_weighted_edges_from(edgelist, weight='len')
        print(' - epsilon graph built.')
        return graph

    def get_epsilon_edges(self): 
        """Extracts 1-skeleton from reduced sets R and E."""
        # Logs
        self.log(self.get_epsilon_edges.__name__, nl=True)
        self.log('epsilon', self.epsilon, sub=True)
        self.log('gamma', self.gamma, sub=True)
        self.log('delta', self.delta, sub=True)
        self.log('data_dim', self.sparseR.shape[-1], sub=True)
        self.log('Rdist_percentile', self.Rdist_percentile, sub=True)
        
        # Extract 1-skeleton of Vietoris-Rips graph
        fn_start = timer()
        vrc = gudhi.RipsComplex(points=np.concatenate([self.sparseR, self.sparseE]),
                                max_edge_length=self.epsilon)
        st = vrc.create_simplex_tree(max_dimension=1)
        fn_vr_end = timer()
        self.log('RE_graph_time_elapsed', timedelta(seconds=fn_vr_end - fn_start), sub=True)
        vrs = st.get_skeleton(1) 
        vrs = list(vrs)
        edgelist = [vrs[i][0] + [vrs[i][1]] for i in range(len(vrs)) if len(vrs[i][0]) > 1]
        self.log('total_time_elapsed', timedelta(seconds=timer() - fn_start), sub=True)
        self.log('skeleton_length', len(vrs), sub=True)
        return edgelist
    
    # -------------------------------------------------------------------------- #
    # GeomCA: evaluation
    # -------------------------------------------------------------------------- #
    def get_graph_consistency(self, num_R_vertices, num_E_vertices):
        """
        Calculates the consistency of the given graph as in Definition 2.2.
        :param num_R_vertices: number of R vertices contained in a graph.
        :param num_E_vertices: number of E vertices contained in a graph.
        :return: consistency score of the graph.
        """
        return 1 - abs(num_R_vertices - num_E_vertices)/(num_R_vertices + num_E_vertices)

    def get_graph_quality(self, RE_graph, R_graph, E_graph):
        """
        Calculates the quality of the given graph as in Definition 2.3,
        :param RE_graph: epsilon graph (or component) containing both R and E points.
        :param R_graph: epsilon graph (or component) restricted to R.
        :param E_graph: epsilon graph (or component) restricted to E.
        :return: quality score of the graph.
        """
        num_R_and_E_edges = R_graph.number_of_edges() + E_graph.number_of_edges()
        num_total_edges = RE_graph.number_of_edges()
        num_RE_edges = (num_total_edges - num_R_and_E_edges)
        # print('Network homogeneous edges: {0}, heterogenerous edges: {1}, total: {2}'.format(
        #     num_R_and_E_edges, num_RE_edges, num_total_edges))
        return num_RE_edges/num_total_edges if num_total_edges != 0 else 0

    def get_network_consistency_and_quality(self):
        """Calculates network consistency and quality as in Definition 2.4,"""
        network_consistency = self.get_graph_consistency(self.R_pts_idx, self.E_pts_idx)
        R_subgraph, E_subgraph = self.extract_subnetworks(self.RE_network)
        network_quality = self.get_graph_quality(self.RE_network, R_subgraph, E_subgraph)
        return network_consistency, network_quality

    def get_connected_components(self):
        """
        Extracts and evaluates graph-connected component of the epsilon graph G^(R U E),
        :returns: dictionaries with component statistics, network statistics and network parameters.
        """
        self.log(self.get_connected_components.__name__, nl=True)
        for comp_idx, comp in enumerate(sorted(nx.connected_components(self.RE_network), key=len, reverse=True)):
            # Split the component into graph restrictions G_i^R and G_i^E.
            subgraph_RE_comp = self.RE_network.subgraph(comp).copy()
            subgraph_R_comp, subgraph_E_comp = self.extract_subnetworks(subgraph_RE_comp)
            self.evaluate_component(comp_idx, subgraph_RE_comp, subgraph_R_comp, subgraph_E_comp)
        self.save_geomCA_logs()
        return self.comp_stats_logger, self.network_stats_logger, self.network_params_logger
    
    def extract_subnetworks(self, graph):
        """
        Extracts graph restrictions H^R and H^E from a given graph H.
        :param graph: epsilon graph (or component) to extract restrictions from.
        :return: graph resticted to R and graph resticted to E.
        """
        R_network = graph.subgraph(np.arange(self.R_pts_idx))
        E_network = graph.subgraph(np.arange(self.R_pts_idx, self.R_pts_idx + self.E_pts_idx))
        return R_network, E_network

    def evaluate_component(self, comp_idx, subgraph_RE_comp, subgraph_R_comp, subgraph_E_comp):
        """
        Evaluates given graph-connected component.
        :param comp_idx: id of the component (in descresing order by their size)
        :param subgraph_RE_comp: epsilon graph of the component containing R and E points..
        :param subgraph_R_comp: epsilon graph of component restricted to R.
        :param subgraph_E_comp: epsilon graph of component restricted to E.
        """
        comp_sparseR_idx = np.array(list(subgraph_R_comp.nodes()))
        comp_sparseE_idx = np.array(list(subgraph_E_comp.nodes())) - self.R_pts_idx
        comp_consistency = self.get_graph_consistency(len(comp_sparseR_idx), len(comp_sparseE_idx))
        comp_quality = self.get_graph_quality(subgraph_RE_comp, subgraph_R_comp, subgraph_E_comp)
        log_msg = 'component {0}, consistency {1}, quality {2}'.format(comp_idx, comp_consistency, comp_quality)
        self.log(log_msg)
        self.comp_stats_logger[comp_idx] = {
            'Ridx': comp_sparseR_idx,
            'Eidx': comp_sparseE_idx,
            'comp_consistency': comp_consistency,
            'comp_quality': comp_quality, 
            'comp_graph': subgraph_RE_comp}

        # Part of global evaluation as in Definition 2.5.
        if comp_consistency > self.comp_consistency_threshold and comp_quality > self.comp_quality_threshold:
            self.network_stats_logger['num_R_points_in_qualitycomp'] += len(comp_sparseR_idx)
            self.network_stats_logger['num_E_points_in_qualitycomp'] += len(comp_sparseE_idx)
        
    # -------------------------------------------------------------------------- #
    # GeomCA: visualization
    # -------------------------------------------------------------------------- #
    def plot_dist_hist(self, distances, filename, epsilon):
        """
        Plots histogram of distances D obtained when estimating epsilon.
        :param distances: array of the calucated distances D..
        :param filename: name of the file to save histrogram to.
        :param epsilon: estimated epsilon value.
        """
        max_ptns = min(len(distances), 10000)
        np.random.shuffle(distances)
        distances = distances[np.random.choice(np.arange(len(distances)), max_ptns)]
        
        plt.figure(1)
        plt.clf()
        avg_dist = np.mean(distances)
        std_dist = np.std(distances)
        plt.hist(distances.reshape(-1), bins=max_ptns)
        _, max_ylim = plt.ylim()
        plt.axvline(avg_dist, color='k', linestyle='dashed', linewidth=1)
        plt.text(avg_dist*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(avg_dist))
        
        plt.axvline(avg_dist - std_dist, color='k', linestyle='dashed', linewidth=1)
        plt.text((avg_dist - std_dist)*1.1, max_ylim*0.7, 
                 'Mean - std: {:.2f}'.format(avg_dist - std_dist))
        
        plt.axvline(epsilon, color='k', linestyle='dashed', linewidth=1)
        plt.text(epsilon *1.1, max_ylim*0.5, 
                 '{1}th percentile: {0:.2f}'.format(epsilon, self.Rdist_percentile))
        
        plt.title('Histogram of {0}, num_pts = {1}'.format(filename, max_ptns))
        path = os.path.join(self.ptns_path, self.save_name + filename)
        plt.savefig(path)
        plt.close()
    
    def project_to_ndim(self, R, E=None, dim=2):
        """
        2D projection using UMAP to visualize higher dimensional reprensetations.
        :param R: array of R points to project.
        :param E: array of E points to project.
        :param dim: dimension of rojected points.
        :return: projected R (and E) points.
        """
        if E is not None:
            if self.RE_2D_projection is not None:
                RE_proj = self.RE_2D_projection.fit_transform(np.concatenate([R, E]))
            else:
                self.RE_2D_projection = umap.UMAP(n_components=dim).fit((np.concatenate([R, E])))
                RE_proj = self.RE_2D_projection.transform(np.concatenate([R, E]))
            R_proj = RE_proj[:len(R), :]
            E_proj = RE_proj[len(R):, :]
            return R_proj, E_proj
        else:
            return umap.UMAP.fit_transform(R)
    
    def plot_ptns(self, R, E, filename, figsize=(5, 5)):
        """"
        Plots given points R and E in 2D.
        :param R: array of R points to visualize.
        :param E: array of E points to visualize.
        :param filename: name of the file to save the plot to.
        :param figsize: size of the plot.
        """
        if R.shape[1] > 2:
            plotR, plotE = self.project_to_ndim(R, E)
        else:
            plotR, plotE = R, E
            
        plt.figure(figsize=figsize)
        plt.scatter(plotR[:, 0], plotR[:, 1], s=30, color=self.R_color, alpha=0.5)
        plt.scatter(plotE[:, 0], plotE[:, 1], s=30, color=self.E_color, alpha=0.5)
        plt.title('{0} point cloud'.format(filename))
        legend_elements = [
            Line2D([0], [0], color=self.R_color, lw=4, label='R'),
            Line2D([0], [0], color=self.E_color, lw=4, label='E')]
        plt.legend(handles=legend_elements)
        
        # Save the plot
        save_path = os.path.join(self.ptns_path, self.save_name + filename)
        plt.savefig(save_path)
        plt.clf()
        plt.close()

    def plot_connected_components(self, figsize=(5, 5)):
        """
        Plots connected components of the obtained epsilon graph build on
        sparsified points projected to 2D. Each color represents one component,
        where E points are marked with X.
        :param figsize: size of the plot.
        """
        if self.sparseR.shape[1] > 2:
            plotR, plotE = self.project_to_ndim(self.sparseR, self.sparseE)
        else:
            plotR, plotE = self.sparseR, self.sparseE
        
        n_components = len(list(self.comp_stats_logger.keys()))
        plt.figure(2, figsize=figsize)
        plt.clf()
        colors = cm.rainbow(np.linspace(0, 1, n_components))
        for i in range(n_components):
            Ridxs = list(self.comp_stats_logger[i]['Ridx'])
            Eidxs = list(self.comp_stats_logger[i]['Eidx'])
            plt.scatter(plotR[Ridxs, 0], plotR[Ridxs, 1], s=30, 
                        color=colors[i], alpha=1.0)
            plt.scatter(plotE[Eidxs, 0], plotE[Eidxs, 1], s=50, marker="X", 
                        color=colors[i], alpha=0.9)
        plt.title('epsilon graph components')
        path = os.path.join(self.ptns_path, self.save_name + 'epsilon_graph_components')
        plt.savefig(path)
        plt.clf()
        plt.close()

    def plot_RE_components_quality(self, annotate_largest=True, min_comp_size=0, display_smaller=False, 
        figsize=(10, 5)):
        """
        Visualizes components quality as a scatter plot.
        :param annotate_largest: if annotate the size (in percentage) of the largest component.
        :param min_comp_size: minimum size (number of vertices) of the components to visualize.
        :param display_smaller: if display aggregated components with size smaller than min_comp_size.
        :param figsize: size of the plot.
        """
        n_comp = len(self.comp_stats_logger.keys()) 
        total_n_pts = self.R_pts_idx + self.E_pts_idx
        max_score, last_display_comp = 0, 0
        small_R_comp, small_E_comp , small_RE_comp = 0, 0, 0
        quality_scores, ticks_labels = [], []
        fig, ax = plt.subplots(figsize=figsize)
        for comp_id in range(n_comp):
            compR = self.comp_stats_logger[comp_id]['Ridx']
            compE = self.comp_stats_logger[comp_id]['Eidx']
            comp_n_points = len(compR) + len(compE)
            if comp_n_points >= min_comp_size:
                comp_quality = np.round(self.comp_stats_logger[comp_id]['comp_quality'], 2)
                max_score = max(max_score, comp_quality)
                last_display_comp = comp_id + 1
                quality_scores.append(comp_quality)
                if len(compR) != 0:
                    if len(compE) != 0:
                        comp_color = 'gray'
                    else:
                        comp_color = self.R_color
                else:
                    comp_color = self.E_color
                
                ax.scatter(comp_id, comp_quality, c=comp_color, linestyle='--',
                        s=1000*(comp_n_points)/total_n_pts, alpha=0.8, zorder=10)
            else:
                if len(compR) != 0:
                    if len(compE) != 0:
                        small_RE_comp += 1
                    else:
                        small_R_comp += 1
                else:
                    small_E_comp += 1
        
        if min_comp_size > 0 and display_smaller:
            if small_RE_comp + small_R_comp + small_E_comp > 0: 
                ticks_labels = [last_display_comp]
                
            if small_RE_comp > 0:
                r = last_display_comp + 2 * len(ticks_labels)
                ticks_labels.append(ticks_labels[-1] + small_RE_comp)
                ax.axvspan(r - 2, r, alpha=0.5, color='gray')
            
            if small_R_comp > 0:
                r = last_display_comp + 2 * len(ticks_labels)
                ticks_labels.append(ticks_labels[-1] + small_R_comp)
                ax.axvspan(r - 2, r, alpha=0.5, color=self.R_color)
                
            if small_E_comp > 0:
                r = last_display_comp + 2 * len(ticks_labels) 
                ticks_labels.append(ticks_labels[-1] + small_E_comp)
                ax.axvspan(r - 2, r, alpha=0.5, color=self.E_color)
            
        # Annotate the largest component
        if annotate_largest:
            largest_comp_size = len(self.comp_stats_logger[0]['Ridx']) + len(self.comp_stats_logger[0]['Eidx'])
            ax.annotate(round(largest_comp_size/total_n_pts, 2), 
                        xy=(0, self.comp_stats_logger[0]['comp_quality'] + 0.03), ha='center', va='bottom', 
                        color='k')
            if max_score == 0:
                ax.plot(0, self.comp_stats_logger[0]['comp_quality'], 'kX')
       
        ax.plot(np.arange(last_display_comp), quality_scores, color='gray', linestyle='--', alpha=0.5, zorder=0)
        displayed_ticks = np.arange(last_display_comp, step=max(int(last_display_comp/10), 1))
        if min_comp_size == 0:
            ax.set_xticks(displayed_ticks)
            ax.set_xticklabels(displayed_ticks)
        else:
            new_ticks = np.arange(last_display_comp, last_display_comp + len(ticks_labels) * 2, 2)
            ax.set_xticks(np.concatenate([displayed_ticks, 
                                          new_ticks]))
            ax.set_xticklabels(list(displayed_ticks) + ticks_labels)
            max_score = 1.0 if max_score == 0 else max_score
        ax.set_ylim((-0.05, max_score + 0.1))
        ax.set_yticks(np.arange(0, max_score + 0.1, 0.1))
        
        # ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel('component index')
        ax.set_ylabel('component quality')
        legend_elements = [
            Line2D([0], [0], markerfacecolor=self.R_color, markersize=10, label='R', marker='o', color='w'),
            Line2D([0], [0], markerfacecolor=self.E_color, markersize=10, label='E', marker='o', color='w'),
            Line2D([0], [0], markerfacecolor='gray', markersize=10, label='mix', marker='o', color='w')]
        ax.legend(handles=legend_elements, ncol=len(legend_elements), 
                  loc='upper center', framealpha=0.5)
        name = 'component_quality_min_size{0}_annotated{1}_displaysmaller{2}'.format(
            min_comp_size, int(annotate_largest), int(display_smaller))
        path = os.path.join(self.results_path, self.save_name + name)
        plt.tight_layout()
        plt.savefig(path)
        plt.clf()
        plt.close()


    def plot_RE_components_consistency(self, annotate_largest=True, min_comp_size=0, display_smaller=False,
        figsize=(10, 5)):
        """
        Visualizes components consistency as a scatter plot.
        :param annotate_largest: if annotate the size (in percentage) of the largest component.
        :param min_comp_size: minimum size (number of vertices) of the components to visualize.
        :param display_smaller: if display aggregated components with size smaller than min_comp_size.
        :param figsize: size of the plot.
        """
        n_comp = len(self.comp_stats_logger.keys()) 
        total_n_pts = self.R_pts_idx + self.E_pts_idx
        max_score, last_display_comp = 0, 0
        small_R_comp, small_E_comp , small_RE_comp = 0, 0, 0
        consistency_scores, ticks_labels = [], []

        fig, ax = plt.subplots(figsize=figsize)
        for comp_id in range(n_comp):
            compR = self.comp_stats_logger[comp_id]['Ridx']
            compE = self.comp_stats_logger[comp_id]['Eidx']
            comp_n_points = len(compR) + len(compE)
            if comp_n_points >= min_comp_size:
                comp_consistency = np.round(self.comp_stats_logger[comp_id]['comp_consistency'], 2)
                max_score = max(max_score, comp_consistency)
                last_display_comp = comp_id + 1
                consistency_scores.append(comp_consistency)
                if len(compR) != 0:
                    if len(compE) != 0:
                        comp_color = 'gray'
                    else:
                        comp_color = self.R_color
                else:
                    comp_color = self.E_color
                
                ax.scatter(comp_id, comp_consistency, c=comp_color, linestyle='--',
                        s=1000*(comp_n_points)/total_n_pts, alpha=0.8, zorder=10)
            else:
                if len(compR) != 0:
                    if len(compE) != 0:
                        small_RE_comp += 1
                    else:
                        small_R_comp += 1
                else:
                    small_E_comp += 1
        
        if min_comp_size > 0 and display_smaller:
            if small_RE_comp + small_R_comp + small_E_comp > 0: 
                ticks_labels = [last_display_comp]
                
            if small_RE_comp > 0:
                r = last_display_comp + 2 * len(ticks_labels)
                ticks_labels.append(ticks_labels[-1] + small_RE_comp)
                ax.axvspan(r - 2, r, alpha=0.5, color='gray')
            
            if small_R_comp > 0:
                r = last_display_comp + 2 * len(ticks_labels)
                ticks_labels.append(ticks_labels[-1] + small_R_comp)
                ax.axvspan(r - 2, r, alpha=0.5, color=self.R_color)
                
            if small_E_comp > 0:
                r = last_display_comp + 2 * len(ticks_labels) 
                ticks_labels.append(ticks_labels[-1] + small_E_comp)
                ax.axvspan(r - 2, r, alpha=0.5, color=self.E_color)
            
        # Annotate the largest component
        if annotate_largest:
            largest_comp_size = len(self.comp_stats_logger[0]['Ridx']) + len(self.comp_stats_logger[0]['Eidx'])
            ax.annotate(round(largest_comp_size/total_n_pts, 2), 
                        xy=(0, self.comp_stats_logger[0]['comp_consistency'] + 0.03), ha='center', va='bottom', 
                        color='k')
            if max_score == 0:
                ax.plot(0, self.comp_stats_logger[0]['comp_consistency'], 'kX')
       
        ax.plot(np.arange(last_display_comp), consistency_scores, color='gray', linestyle='--', alpha=0.5, zorder=0)
        displayed_ticks = np.arange(last_display_comp, step=max(int(last_display_comp/10), 1))
        if min_comp_size == 0:
            ax.set_xticks(displayed_ticks)
            ax.set_xticklabels(displayed_ticks)
        else:
            new_ticks = np.arange(last_display_comp, last_display_comp + len(ticks_labels) * 2, 2)
            ax.set_xticks(np.concatenate([displayed_ticks, 
                                          new_ticks]))
            ax.set_xticklabels(list(displayed_ticks) + ticks_labels)
            max_score = 1.0 if max_score == 0 else max_score
        ax.set_ylim((-0.05, max_score + 0.1))
        ax.set_yticks(np.arange(0, max_score + 0.1, 0.1))
        
        # ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel('component index')
        ax.set_ylabel('component consistency')
        legend_elements = [
            Line2D([0], [0], markerfacecolor=self.R_color, markersize=10, label='R', marker='o', color='w'),
            Line2D([0], [0], markerfacecolor=self.E_color, markersize=10, label='E', marker='o', color='w'),
            Line2D([0], [0], markerfacecolor='gray', markersize=10, label='mix', marker='o', color='w')]
        ax.legend(handles=legend_elements, ncol=len(legend_elements), 
                  loc='upper center', framealpha=0.5)
        name = 'component_consistency_min_size{0}_annotated{1}_displaysmaller{2}'.format(
            min_comp_size, int(annotate_largest), int(display_smaller))
        path = os.path.join(self.results_path, self.save_name + name)
        plt.tight_layout()
        plt.savefig(path)
        plt.clf()
        plt.close()
            
    
