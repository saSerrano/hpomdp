/**
 * \class HPomdp
 * 
 * \brief This class implements the architecture proposed in the article "Knowledge-Based Hierarchical POMDPs for Task Planning".
 * 
 * \author $Author: Sergio A. Serrano$
 * 
 * \date $Date: 02/05/2020$
 * 
 * Contact: sserrano@inaoep.mx
 */

//Standard
#include <iostream>
#include <fstream>
#include <exception>
#include <time.h>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>
#include <iomanip>
#include <chrono>
#include <tuple>
#include <map>
#include <cmath>

//OpenCV for ploting hierarchicalexecution traces
#include <opencv2/opencv.hpp>

//JSON
#include <nlohmann/json.hpp>

//AIToolbox
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Algorithms/PBVI.hpp>
#include <AIToolbox/POMDP/Policies/Policy.hpp>
#include <AIToolbox/Impl/CassandraParser.hpp>

//For Handling tree structures
#include <TreeHandle.hpp>

//For handling neighbor
#include <Neighborhood.hpp>

//For generating an environment for the MICAI experiments
#include <EnvGen.hpp>

using namespace AIToolbox;
using json = nlohmann::json;
using namespace std;
using namespace cv;

#ifndef H_POMDP
#define H_POMDP

//Probability matrix of a single action. This typedef can be used to represent
//both T and O functions, where:
// - Vector of starting states (for T) or ending state (for O)
// - Vector of ending states or perceived observation
// - Vector of probabilities of transitions or perceiving observations
typedef tuple<vector<string>, vector<string>, vector<double> > pMat;

//POMDP
// - Cassandra POMDP model
// - POMDP's list of states
// - POMDP's list of actions
// - POMDP's list of observations
// - POMDP's Transition function
// - POMDP's Observation function
// - POMDP's list of goal states
// - POMDP's list of goal starting_states-action pairs
// - POMDP's list of punishment states
// - POMDP's list of punishment starting_states-action pairs
// - POMDP's horizon for which it will be solved
typedef tuple<POMDP::Model<MDP::Model>,
	      vector<string>,
	      vector<string>,
	      vector<string>,
	      vector<pMat>,
	      vector<pMat>,
	      vector<string>,
	      vector<tuple<string,string> >,
	      vector<string>,
	      vector<tuple<string,string> >,
	      unsigned
	      > pomdp;

//POMDP Policy
// - POMDP solved policy
// - List of actions it can invoke
// - List of observations it can perceive with its actions
// - List of states
// - List of abstract states (associated to states of the previous list)
// - POMDP model
typedef tuple<POMDP::Policy,vector<string>,vector<string>,vector<string>,vector<string>,POMDP::Model<MDP::Model> > policy;

class HPomdp
{	
	private:

		TreeHandle hs_;// Hierarchy of state space
		Neighborhood nh_;// Spatial configuration of the original state space
		map<string,policy> AA_;// List of abstract actions

		//Control flags
		bool env_loaded_;
		bool sim_initialized_;
		bool sim_state_set_;

		//POMDPs at every level
		vector< vector<string> > S_;
		vector< vector<string> > A_;
		vector< vector<string> > Z_;
		vector< vector<pMat> > T_;
		vector< vector<pMat> > O_;

		//Vector of POMDP models of every level
		vector<POMDP::Model<MDP::Model> > M_;

		//Mapping from z to the states from which it is observable
		map<string,vector<string>> z2s_;

		//-----------------------------------------
		//H-POL exec variables
		//Global belief vector for execution of hierarchical policy
		vector<map<string,double> > B_;
		//Map for updating the concrete belief vector dynamically
		map<string,double> b_t_;
		//Hierarchical policy (result of the 'hPolExec' method)
		vector<policy> HP_;
		//-----------------------------------------

		//Vectors for hierarchical execution
		vector<string> xB_;
		vector< tuple<string,string> > xAZ_;
		string ws_state;
		int ca_count_;// Concrete actions counter

		//Keep record of a full hierarchical trace
		int top_lvl_;// upper-most level at which actions are executed
		vector<vector<string> > trace_a;
		vector<vector<string> > trace_z;
		vector<vector<vector<double> > > trace_b;
		vector<vector<vector<string> > > trace_is;
		vector<string> trace_s;

		//World simulator model
		vector<pMat> ws_T;
		vector<pMat> ws_O;

		//Specific for navigation problem
		vector<string> concrete_A;
		vector<double> concrete_T;
		vector<double> concrete_O;
		vector< tuple<int,int,double> > o_dist;

		//The most likely state (for TLP)
		string mls;

		//Flag for when HP gets stucked between 2 local policies
		bool stuck;

		/**
		 * \brief This method is for loading the hierarchical & spatial description of a navigation environment, as well as the concrete probabilities for the T & O functions.
		 * \param env_file Path to the JSON file containing the environmet's hierarchical & spatial description.
		 * \param prob_file Path to the JSON file containing the concrete probabilities.
		 * \return Returns the true if both files were successfully loaded, otherwise false.
		*/
		bool loadEnv(string const &env_file, string const &prob_file);

		/**
		 * \brief This method generates the full POMDP model for the hierarchy's bottom level, that is, at the concrete level, such model is constituted by S,A,Z,T,O, whereas teh reward function is dynamically defined at the moment abstract actions are built.
		*/
		void generateBottomModel();

		/**
		 * \brief This method generates, for the bottom model, a mapping from each observation to the state from which it can be observed.
		*/
		void generateBottomZSMap();

		/**
		 * \brief This method
		 * \param S Vector of strings that represent POMDP's states.
		 * \param A Vector of strings that represent POMDP's actions.
		 * \param Z Vector of strings that represent POMDP's observations.
		 * \param T Vector of transition probability cubes of each action.
		 * \param O Vector of observation probability cubes of each action.
		 * \param G Vector of strings that represent POMDP's goal states.
		 * \param Gpair Vector of tuples holding a pair of strings that represent POMDP's starting_state-action pairs that should have a goal reward.
		 * \param P Vector of strings that represent POMDP's punishment states.
		 * \param Ppair Vector of tuples holding a pair of strings that represent POMDP's starting_state-action pairs that should have a pinishement reward.
		 * \param discount Value used in the POMDP's solving process, should be in the interval [0,1].
		 * \param succ Reference to a flag that will hold the success status of building the POMDP.
		 * \return Returns a 'pomdp' typedef that holds the POMDP
		*/
		pomdp generatePomdp(vector<string> const &S,vector<string> const &A,vector<string> const &Z,vector<pMat> const &T,vector<pMat> const &O,vector<string> const &G,vector<tuple<string,string> > const &Gpair,vector<string> const &P,vector<tuple<string,string> > const &Ppair,double const &discount,vector<string> const &rewOrder,bool &succ);

		/**
		 * \brief This method builds the hierarchy of abstract actions for the hierarchical representation of the state-space (stored in 'hs_'). It starts by using the concrete-level model to construct abstract  actions for the immediate level above in the hierarchy, and it repeats for the next level, and so on until it reaches the top of the hierarchy.
		 * \param n_sim_aa_params Amount of times each abstract action will be simulated to estimate its upper T & O probabilities.
		*/
		void buildHierarchyAA(int const &n_sim_aa_params);

		/**
		 * \brief This method for evaluating if the POMDP described by the input arguments is consistent (the distribution prob. function for every state-action pair sums 1, and that every element in S and Z is in every probability matrix of T and O).
		 * \param S List of states.
		 * \param A List of actions.
		 * \param Z List of observations.
		 * \param T Transition probability matrices for actions in A.
		 * \param O Observation probability matrices for actions in A.
		 * \return Returns true if the checked POMDP model passed all revisions.
		*/
		bool checkPomdp(vector<string> const &S,vector<string> const &A,vector<string> const &Z,vector<pMat> const &T,vector<pMat> const &O);

		/**
		 * \brief This method is for generating the T & O probabilities for starting state "S0" and abstract action modeled by policy "pol".
		 * \param p A tuple containing a POMDP's parameters.
		 * \param S0 Starting abstract state.
		 * \param S1 List non-starting abstract states.
		 * \param S1_peri List of the non-starting abstract states (as they are named within the model in "p").
		 * \param pol A tuple containing a POMDP's solved policy and the model's list of actions and observations.
		 * \param n_sim_aa_params Amount of times each abstract action will be simulated to estimate its upper T & O probabilities.
		 * \param gv_lvl Index (in the global vectors) of the level over which the abstract action is modeled.
		 * \return Returns a tuple containing the T & O probabilities for starting state "S0" and abstract action modeled with policy "pol".
		*/
		tuple<pMat,pMat> modelTO(pomdp const &p,string const &S0,vector<string> const &S1,vector<string> const &S1_peri,policy const &pol,int const &n_sim_aa_params,unsigned const &gv_lvl);

		/**
		 * \brief This method is for generate the transition probabilities of a policy, so it can be used as an abstract action. The states in model "p" are relabeled as abstract states. Each peripheral state is an asbtract state, while the remaining states are clustered into a single abstract state. All the transitions whose probabilities are ebing computed, start in the non-peripheral abstract state.
		 * \param p A tuple containing a POMDP's parameters.
		 * \param pol A tuple containing a POMDP's solved policy and the model's list of actions and observations.
		 * \param peri A vector containing all the peripheral states of the model described in "p".
		 * \param sim_runs The amount of simulations to be ran on policy "pol" and model "p".
		 * \param gv_lvl Index (in the global vectors) of the level over which the abstract action is modeled.
		 * \return Returns the transition probabilities that start in a non-peripheral state.
		*/
		pMat simPolicy(pomdp const &p,policy const &pol,vector<string> const &peri,unsigned int const &sim_runs,unsigned const &gv_lvl);

		/**
		 * \brief This method is for truncating the precision of a double variable (without rounding the least significant digit) to a given amount of decimal digits, weather they all are 0's or not.
		 * \param d The double variable to be truncated.
		 * \param dec_prec Amount of digits, at the right of the decimal point, to be kept after truncating.
		 * \return Returns the truncated value.
		*/
		double truncate(double const &d,unsigned const &dec_prec);

	public:

		/**
		 * \brief Default constructor that initializes without any parameters. Intances initialized with this constructor have no use.
		*/
		HPomdp();

		/**
		 * \brief Constructor method that initializes an instance using description files of the environment's hierarchical & spatial configuration, as well as the problem's concrete probabilities for the T & O functions.
		 * \param env_file Path to the JSON file containing the environmet's hierarchical & spatial description.
		 * \param prob_file Path to the JSON file containing the concrete probabilities.
		*/
		HPomdp(string const &env_file, string const &prob_file);

		/**
		 * \brief Constructor method that initializes an instance using description files of the environment's hierarchical & spatial configuration, as well as the problem's concrete probabilities for the T & O functions.
		 * \param env_file Path to the JSON file containing the environmet's hierarchical & spatial description.
		 * \param prob_file Path to the JSON file containing the concrete probabilities.
		 * \param n_sim_aa_params Amount of time to simulate each abstract action in order to estimate its upper T & O probabilities.
		*/
		HPomdp(string const &env_file, string const &prob_file,int const &n_sim_aa_params);

		/**
		 * \brief This method is for simulating an environment using the transition & observation probabilities from vectors of pMats 'ws_T' & 'ws_O', respectively.
		 * \param a Concrete action to be executed in the simulator.
		 * \return Returns the concrete observation perceived after executing action 'a'.
		*/
		string interactWithWorld(string const &a);

		/**
		 * \brief This method is for updating the agent's belief vector at the concrete level, but in a dynamic way in which only the states with non-zero probability are part in the vector, and in each update, states can be removed or added. The belief vector is stored in the class' variable 'b_t_'.
		 * \param a Executed action.
		 * \param z Perceived observation after executing action 'a'.
		*/
		void updateBeliefDyn(string const &a, string const &z);

		/**
		 * \brief This method executes the shell command that is passed as input parameter.
		 * \param cmd String to be executed in console.
		 * \return Returns the text generated by the OS as result of executing the command held in cmd.
		*/
		std::string shellCmd(std::string cmd);

		/**
		 * \brief This method is for checking if the instance has been properly initialized by loading its configuration files.
		 * \return Returns true if the environment's files were already successfully loaded, otherwise false.
		*/
		bool envLoaded() const;

		/**
		 * \brief This method is for generating a concrete-probabilities file (which is required in order to build a hierarchy  of abstract actions).
		 * \param file Name of the output JSON file that store the concrete probabilities.
		 * \param T_prec Probability that action will of actually reaching their intended end state.
		 * \param O_prec Probability of perceive the correct observation.
		 * \param O_kernel_size Dimensions of the discrete Gaussian kernel.
		 * \param sd Standard deviation for the Gaussian kernel.
		 * \return Returns the true if the file was successfully written, otherwise false.
		*/
		bool createProbFile(string const &file,double const &T_prec, double const &O_prec, int O_kernel_size = -1, double sd = 1.0);

		/**
		 * \brief This method is for setting the transition and observation model to be simulated, if no probability file is provided, then the model for the current bottom level is use instead.
		 * \param prob_file A file that contains the transition and observation probabilities for he environment currently loaded.
		 * \return Returns true if a model was succesfully set, otherwise, false.
		*/
		bool setSimulatorModel(string const &prob_file = string("use-concrete-model"));

		/**
		 * \brief This method is for  setting the world-simulator to a given state.
		 * \param world_state The state of the world-simulator to be set.
		 * \return Returns true if the input state is a valid one, otherwise false.
		*/
		bool setSimulatorState(string const &world_state);

		/**
		 * \brief This method is for queuring the current state in the world-simulator.
		 * \return Returns the state of the world-simulator.
		*/
		string getSimulatorState();

		/**
		 * \brief This method is for reseting the counter of concrete actions executed.
		*/
		void resetStepCounter();

		/**
		 * \brief This method is for queuring the amount of concrete actions executed since the last time the counter was reset.
		 * \return Returns the count of concrete actions executed.
		*/
		int getStepCounter();

		/**
		 * \brief This method decomposes a string into a vector of substring that are delimited by 'delimiter'.
		 * \param fullString String to be decomposed.
		 * \param delimiter String that represents the delimiter.
		 * \return Vector of strings.
		*/
		vector<std::string> splitStr(std::string const &fullString,std::string const &delimiter);

		/**
		 * \brief This method computes the average and standard deviation of a vector of values 'vec'.
		 * \param vec Vector of values.
		 * \return Tuple that holds <average,standard deviation>.
		*/
		tuple<double,double> avgSd(vector<double> const &vec);

		/**
		 * \brief This method is for computing the time between to timestamps ('t1' and 't2') in seconds.
		 * \param t1 First timestamp.
		 * \param t2 Second timestamp.
		 * \return Time between 't1' and 't2', expressed in seconds.
		*/
		double u2s(std::chrono::time_point<std::chrono::high_resolution_clock> t1,std::chrono::time_point<std::chrono::high_resolution_clock> t2);

		/**
		 * \brief This method computes a policy for a POMDP 'p' with the Point-Based Value Iteration (PBVI) algorithm.
		 * \param p POMDP model.
		 * \param nBeliefs Amount of belief points that PBVI will use to compute the policy.
		 * \param horizon Horizon that PBVI will use to compute the policy.
		 * \param epsilon Error margin that represents the convergence of PBVI.
		 * \return POMDP policy.
		*/
		policy solveModel(pomdp &p,size_t nBeliefs,unsigned horizon,double epsilon);

		/**
		 * \brief This method builds POMDP policies to transit between subregions of the state space that are neighbors.
		 * \return List of POMDP policies.
		*/
		map<string,policy> srPolicies();

		/**
		 * \brief This method builds the POMDP that models the environment at the bottom level of the hierarchy of states.
		 * \return POMDP of the concrete level.
		*/
		pomdp concModel();

		/**
		 * \brief This method is for computing the POMDP policy to reach the 'goal_s' within the subregion of the state space that contains 'goal_s'. This method is employed by TLP  to compute the policy in the subregion  of the state space that contains the goal state.
		 * \param final_sr Name of the subregion of the state space that contains 'goal_s'.
		 * \param goal_s Goal state.
		 * \return POMDP policy.
		*/
		policy finalPol(string const &final_sr,string const &goal_s);

		/**
		 * \brief This method is for executing a POMDP policy.
		 * \param P POMDP policy.
		 * \param max_s Maximum amoun of steps that are allowed before the policy is stopped.
		 * \param s0 Initial state, if it is not known a uniform distribution will be used.
		 * \return Amount of states at index 'i'.
		*/
		tuple<string,int> exePol(policy P,int max_s,string s0 = string("not-state"));

		/**
		 * \brief This method is for writing a summarize vector that contains the average and standard deviation of the samples in the v1-v6 vectors.
		 * \param v1 Vector of successful runs.
		 * \param v2 Vector of steps taken.
		 * \param v3 Vector of path relative costs.
		 * \param v4 Vector of manhattan errors.
		 * \param v5 Vector of iterative planning time (seconds).
		 * \param v6 Vector of relative errors (M.E. / Opt. path ratio).
		 * \param v7 Vector of shortest length for each pair of initial-goal states.
		 * \param var_name Name of the variable that is control variable in this experiment.
		 * \param var_value Value of the variable that is control variable in this experiment.
		 * \param f_name Name of the output file.
		 * \return Returns true if the summarize vector was successfully written.
		*/
		bool saveSummFile(vector<double> const &v1,vector<double> const &v2,vector<double> const &v3,vector<double> const &v4,vector<double> const &v5,vector<double> const &v6,vector<double> const &v7,string const &var_name,double const &var_value,string const &f_name);

		/**
		 * \brief This method sets the belief state distribution at the bottom level of the hierarchy of states as a uniform distribution.
		*/
		void uncertainS();

		/**
		 * \brief This method is for querying the amount of states there are in the hierarchy of states at the index 'i'.
		 * \param i Index in the hierarchy of states, where 'i=0' is the bottom level.
		 * \return Amount of states at index 'i'.
		*/
		int getSizeS(int const &i);

		/**
		 * \brief This method is for computing the dot product between a belief state distribution and an alpha-vector from a value function.
		 * \param b Belief state distribution.
		 * \param a Alpha vector.
		 * \return Returns the dot product betwen 'b' and 'a'.
		*/
		double dotP(POMDP::Belief const &b, Eigen::Matrix<double,Eigen::Dynamic,1> const &a);

		/**
		 * \brief This method is for computing a hierarchical policy to reach 'gs'.
		 * \param gs Goal state.
		 * \param debug Flag that indicates if information of the construction of the hierarchical policy should be displayed.
		 * \return Returns false if 'gs' does not exist in the hierarchy of states, otherwise, true.
		*/
		bool hPolPlan(string const &gs,bool const &debug);

		/**
		 * \brief This method is for executing the hierarchical policy.
		 * \param max_steps Maximum amount of concrete steps allowed before the hierarchical policy is stopped.
		 * \param debug Flag that indicates if information of the execution process should be displayed.
		 * \return Returns true if the hierarchical policy was successfully executed, otherwise, false.
		*/
		bool hPolExec(int const &max_steps,bool const &debug);

		/**
		 * \brief This method is for executing both, local policies of a hierarchical policy, or policies of abstract actions that are in the current hierarchy of actions. If
		 * \param a Name of the action to be executed (abstract or concrete, but it must exist in the hierarchy of actions).
		 * \param lp_id The index in the global vector HP_ of the local policy to be executed.
		 * \param debug Flag that is set, debug info will be diplayed in console while executing.
		 * \return Returns an observation if a concrete action was executed, the last action executed if a local policy or an abstract was executed, and an empty string if something went wrong.
		*/
		string execPA(string const &a,int const &lp_id,int const &max_steps,bool const &debug);

		/**
		 * \brief This method is for drawing an action from a policy's value function 'vf', after modifying it with the entropy-based weight (as shown in Algorithm 5).
		 * \param vf Value function of the original policy (before the entropy-based weight is applied).
		 * \param b Current state belief distribution.
		 * \param t Remaining steps in the execution loop.
		 * \param extra_idx Index of the external state.
		 * \param se Shannon entropy of the external state's belief probability.
		 * \param max_se Maximum possible Shannon entropy the external state can have.
		 * \return Returns the index of the best action.
		*/
		size_t localSampleAction(POMDP::ValueFunction const &vf,POMDP::Belief const &b,int const &t,int const &extra_idx,double const &se,double const &max_se);

		/**
		 * \brief This method is for computing the Shannon entropy of a set of elements.
		 * \param p Vector containing the probability of element  that will be considered in the computation.
		 * \return Returns the Shannon entropy value of 'p'.
		*/
		double entropy(vector<double> const &p);

		/**
		 * \brief This method is for executing generating a navigation environment and run the FP, TLP and HP planners that were evaluated in the article "Knowledge-Based Hierarchical POMDPs for Task Planning".
		 * \param argc Amount of input arguments.
		 * \param argv Array of input arguments.
		 * \return Returns true if the input arguments are valid to execute the experiments, otherwise, false.
		*/
		bool run(int argc,char** argv);
};

#endif
