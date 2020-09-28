#include <HPomdp.hpp>

using namespace std;
using namespace cv;
using namespace AIToolbox;
using json = nlohmann::json;

//Exception designed to catch non-modeled observations perceived during execution.
class excO : public std::exception
{
	private:

		string obs;

	public:

		excO()
		{
			obs = "";
		}

		excO(string const &obs)
		{
			this->obs = obs;
		}

		const char* what() const throw ()
		{
			string e_case;
			e_case = "Perceived a not-modeled observation";
			if(obs != "") e_case += (": "+obs);

			return e_case.c_str();
		}
};

//Exception designed to catch perceived observations that have 0 prob.
//of being perceived after a given action
class excAO : public std::exception
{
	private:

		string act;
		string obs;

	public:

		excAO()
		{
			act = "";
			obs = "";
		}

		excAO(string const &act,string const &obs)
		{
			this->act = act;
			this->obs = obs;
		}

		const char* what() const throw ()
		{
			string e_case;
			if(act != "" && obs != "")
			{
				e_case = "Obs. "+obs+" was perceived after action "+act;
				e_case += ", but has 0 prob. of being perceived in the model.";
			}
			else
			{
				e_case = "An obs. was perceived, that has 0 prob. of being perceived after the last action executed.";
			}

			return e_case.c_str();
		}
};

HPomdp::HPomdp()
{
	env_loaded_ = false;
	sim_initialized_ = false;
	sim_state_set_ = false;

	//seed random number generator
	srand(time(NULL));
}

HPomdp::HPomdp(string const &env_file, string const &prob_file)
{
	env_loaded_ = false;
	sim_initialized_ = false;
	sim_state_set_ = false;

	//seed random number generator
	srand(time(NULL));

	//Load the environment and concrete probabilities
	bool res = loadEnv(env_file,prob_file);

	if(res)
	{
		//Generate the POMDP parameters for the hierarchy's bottom level
		generateBottomModel();

		//Generate the mapping from observations to states at the bottom model
		generateBottomZSMap();

		//Build the hierarchy of abstract actions
		int n_sim_aa_params(2);
		buildHierarchyAA(n_sim_aa_params);
	}
}

HPomdp::HPomdp(string const &env_file, string const &prob_file,int const &n_sim_aa_params)
{
	env_loaded_ = false;
	sim_initialized_ = false;
	sim_state_set_ = false;

	//seed random number generator
	srand(time(NULL));

	//Load the environment and concrete probabilities
	bool res = loadEnv(env_file,prob_file);

	if(res)
	{
		//Generate the POMDP parameters for the hierarchy's bottom level
		generateBottomModel();

		//Generate the mapping from observations to states at the bottom model
		generateBottomZSMap();

		//Build the hierarchy of abstract actions
		buildHierarchyAA(n_sim_aa_params);
	}
}

bool HPomdp::envLoaded() const
{
	return env_loaded_;
}

bool HPomdp::loadEnv(string const &env_file, string const &prob_file)
{
	//Check for a valid file
	ifstream infile(prob_file);
	if(!infile.is_open())
	{
		cout << "ERROR[loadEnv]: could not open " + prob_file << endl;
		env_loaded_ = false;
		return false;
	}

	//Load the probabilities file
	try
	{
		//Parse the file into a JSON structure
		json jj;
		infile >> jj;
		infile.close();

		//List of concrete actions that operate at the bottom level in the hierarchy
		vector<string> concrete_actions;
		concrete_actions.push_back("up");
		concrete_actions.push_back("down");
		concrete_actions.push_back("left");
		concrete_actions.push_back("right");
		concrete_A = concrete_actions;

		//Clear the concrete probability vectors
		concrete_T.clear();
		concrete_O.clear();
		o_dist.clear();

		//Gather the transition and observation probabilities
		for(unsigned int i = 0; i < jj["Function"].size(); i++)
		{
			vector<double> tmp;
			vector< tuple<int,int,double> > o_dist_tmp;

			//Gather the transition probabilities
			if(jj["Function"][i]["type"] == "T")
			{
				tmp.push_back(static_cast<double>(jj["Function"][i]["up"]));
				tmp.push_back(static_cast<double>(jj["Function"][i]["down"]));
				tmp.push_back(static_cast<double>(jj["Function"][i]["left"]));
				tmp.push_back(static_cast<double>(jj["Function"][i]["right"]));

				concrete_T = tmp;
			}

			//Gather the observation probabilities
			if(jj["Function"][i]["type"] == "O")
			{
				tmp.push_back(static_cast<double>(jj["Function"][i]["precision"]));

				concrete_O = tmp;
			}

			//Gather the observation probabilities as a distribution function
			if(jj["Function"][i]["type"] == "O-dist")
			{
				for(unsigned int j = 0; j < jj["Function"][i]["p-dist"].size(); j++)
				{
					int c_x = jj["Function"][i]["p-dist"][j]["x"];
					int c_y = jj["Function"][i]["p-dist"][j]["y"];
					double c_p = jj["Function"][i]["p-dist"][j]["prob"];
					tuple<int,int,double> c_tmp(c_x,c_y,c_p);
					o_dist_tmp.push_back(c_tmp);
				}

				o_dist = o_dist_tmp;
			}
		}

		//Check for valid values
		if(concrete_T.size() != 4 || concrete_O.size() != 1)
		{
			env_loaded_ = false;
			return false;
		}
		for(unsigned int i = 0; i < concrete_T.size(); i++)
		{
			bool fail(false);
			if(concrete_T[i] < 0 || concrete_T[i] > 1) fail = true;
			if(i == 0)
			{
				if(concrete_O[i] < 0 || concrete_O[i] > 1) fail = true;
			}

			if(fail)
			{
				env_loaded_ = false;
				return false;
			}
		}
		for(unsigned int i = 0; i < o_dist.size(); i++)
		{
			double cell_prob = get<2>(o_dist[i]);
			if(cell_prob < 0 || cell_prob > 1)
			{
				env_loaded_ = false;
				return false;
			}
		}
	}
	catch(std::exception &e)
	{
		cout << e.what() << endl;
		env_loaded_ = false;
		return false;
	}

	//Load the environment file
	bool hs_res = hs_.navFromJson(env_file);
	bool nh_res = nh_.navFromJson(env_file);

	//Check if the environment was successfully loaded
	if(!hs_res || !nh_res)
	{
		env_loaded_ = false;
		return false;
	}
	else
	{
		//Propagate the spatial relations all the way in the hierarchical representation
		nh_.propNeigNav(hs_);

		env_loaded_ = true;
		return true;
	}
}

bool HPomdp::checkPomdp(vector<string> const &S,vector<string> const &A,vector<string> const &Z,vector<pMat> const &T,vector<pMat> const &O)
{
	ofstream outf("pomdp-debug.txt");
	outf << "---------------------  CHECKING POMDP PARAMETERS TO BE VALID  ---------------------" << endl;
	if(S.size() == 0) outf << ">> There are no states." << endl;
	if(A.size() == 0) outf << ">> There are no actions." << endl;
	if(Z.size() == 0) outf << ">> There are no observations." << endl;
	if(A.size() != T.size()) outf << ">> The amount of actions differs from the amount of transition-prob matrices." << endl;
	if(A.size() != O.size()) outf << ">> The amount of actions differs from the amount of observation-prob matrices." << endl;

	outf << "---------------------  CHECK FOR NON-REPEATED ELEMENTS:S,A,Z  ---------------------" << endl;
	//Check for repeated elements in S
	vector<string> rep;
	vector<string>::iterator ite2;
	for(unsigned int i = 0; i < S.size(); i++)
	{
		ite2 = find(rep.begin(),rep.end(),S[i]);
		if(ite2 != rep.end()) continue;
		vector<string>::const_iterator ite;
		ite = find(S.begin()+i+1,S.end(),S[i]);
		if(ite != S.end()) rep.push_back(S[i]);
	}
	if(rep.size() > 0)
	{
		outf << ">> The following states are repreated in S: ";
		for(unsigned int i = 0; i < rep.size(); i++)
		{
			if(i > 0) outf << ", ";
			outf << rep[i];
		}
		outf << endl;
	}

	//Check for repeated elements in A
	rep.clear();
	for(unsigned int i = 0; i < A.size(); i++)
	{
		ite2 = find(rep.begin(),rep.end(),A[i]);
		if(ite2 != rep.end()) continue;
		vector<string>::const_iterator ite;
		ite = find(A.begin()+i+1,A.end(),A[i]);
		if(ite != A.end()) rep.push_back(A[i]);
	}
	if(rep.size() > 0)
	{
		outf << ">> The following states are repreated in A: ";
		for(unsigned int i = 0; i < rep.size(); i++)
		{
			if(i > 0) outf << ", ";
			outf << rep[i];
		}
		outf << endl;
	}

	//Check for repeated elements in Z
	rep.clear();
	for(unsigned int i = 0; i < Z.size(); i++)
	{
		ite2 = find(rep.begin(),rep.end(),Z[i]);
		if(ite2 != rep.end()) continue;
		vector<string>::const_iterator ite;
		ite = find(Z.begin()+i+1,Z.end(),Z[i]);
		if(ite != Z.end()) rep.push_back(Z[i]);
	}
	if(rep.size() > 0)
	{
		outf << ">> The following states are repreated in Z: ";
		for(unsigned int i = 0; i < rep.size(); i++)
		{
			if(i > 0) outf << ", ";
			outf << rep[i];
		}
		outf << endl;
	}

	outf << "---------------------  CHECK FOR VALID & COMPLETE T MATRICES  ---------------------" << endl;

	//Check for complete and valid transition probability matrices
	for(unsigned int i = 0; i < T.size(); i++)
	{
		vector<string>::const_iterator ite;
		vector<string> s0 = get<0>(T[i]);
		vector<string> s1 = get<1>(T[i]);
		vector<double> tp = get<2>(T[i]);
		for(unsigned int j = 0; j < S.size(); j++)
		{
			double sa_dist(0.0);
			vector<string> sa_s1;
			vector<double> sa_p;
			for(unsigned int k = 0; k < s0.size(); k++)
			{
				if(s0[k] == S[j])
				{
					sa_dist += tp[k];
					sa_s1.push_back(s1[k]);
					sa_p.push_back(tp[k]);
				}
			}

			//Truncate the precision of the difference between the cumulative prob. and 1
			//in order to avoid precision errors during comparison
			double diff = std::abs(static_cast<double>(1.0) - sa_dist);
			diff = truncate(diff,6);

			//An incomplete distribution for the s-a pair
			if(diff != double(0.0))
			{
				outf << ">> The p-distribution for the T pair (" + S[j] + ", " + A[i] + ") sums " << sa_dist << " & it must be 1." << endl;
				outf << ">> Individual probs: ";
				for(unsigned int k = 0; k < sa_p.size(); k++) outf << sa_p[k] << " ";
				outf << endl;
			}

			//Check that there are no repeated ending states
			bool repeated(false);
			for(unsigned int k = 0; k < sa_s1.size(); k++)
			{
				ite = find(sa_s1.begin()+k+1,sa_s1.end(),sa_s1[k]);
				if(ite != sa_s1.end())
				{
					repeated = true;
					break;
				}
			}
			if(repeated)
			{
				outf << ">> The p-distribution for the T pair (" + S[j] + ", " + A[i] + ") has repeated ending states" << endl;
				outf << ">> Ending states: ";
				for(unsigned int k = 0; k < sa_s1.size(); k++) outf << sa_s1[k] + " ";
				outf << endl;
			}
		}
	}

	outf << "---------------------  CHECK FOR VALID & COMPLETE O MATRICES  ---------------------" << endl;

	//Check for complete and valid observation probability matrices
	for(unsigned int i = 0; i < O.size(); i++)
	{
		vector<string>::const_iterator ite;
		vector<string> s0 = get<0>(O[i]);
		vector<string> ob = get<1>(O[i]);
		vector<double> op = get<2>(O[i]);
		for(unsigned int j = 0; j < S.size(); j++)
		{
			double sa_dist(0.0);
			vector<string> sa_ob;
			vector<double> sa_p;
			for(unsigned int k = 0; k < s0.size(); k++)
			{
				if(s0[k] == S[j])
				{
					sa_dist += op[k];
					sa_ob.push_back(ob[k]);
					sa_p.push_back(op[k]);
				}
			}

			//Truncate the precision of the difference between the cumulative prob. and 1
			//in order to avoid precision errors during comparison
			double diff = std::abs(static_cast<double>(1.0) - sa_dist);
			diff = truncate(diff,6);

			//An incomplete distribution for the s-a pair
			if(diff != double(0.0))
			{
				outf << ">> The p-distribution for the O pair (" + S[j] + ", " + A[i] + ") sums " << sa_dist << " & it must be 1." << endl;
				outf << ">> Individual probs: ";
				for(unsigned int k = 0; k < sa_p.size(); k++) outf << sa_p[k] << " ";
				outf << endl;
			}

			//Check that there are no repeated ending states
			bool repeated(false);
			for(unsigned int k = 0; k < sa_ob.size(); k++)
			{
				ite = find(sa_ob.begin()+k+1,sa_ob.end(),sa_ob[k]);
				if(ite != sa_ob.end())
				{
					repeated = true;
					break;
				}
			}
			if(repeated)
			{
				outf << ">> The p-distribution for the O pair (" + S[j] + ", " + A[i] + ") has repeated observations" << endl;
				outf << ">> Observations: ";
				for(unsigned int k = 0; k < sa_ob.size(); k++) outf << sa_ob[k] + " ";
				outf << endl;
			}
		}
	}

	outf << "------------ CHECK THAT ALL STATES & OBSERVATIONS IN T & O ARE IN S & Z -----------" << endl;

	//Check if all states from the T-function are in S
	for(unsigned int i = 0; i < T.size(); i++)
	{
		vector<string>::const_iterator ite;
		vector<string> s0 = get<0>(T[i]);
		vector<string> s1 = get<1>(T[i]);
		for(unsigned int j = 0; j < s0.size(); j++)
		{
			bool no_s0(false);
			bool no_s1(false);
			ite = find(S.begin(),S.end(),s0[j]);
			if(ite == S.end()) no_s0 = true;
			ite = find(S.begin(),S.end(),s1[j]);
			if(ite == S.end()) no_s1 = true;

			//Display missing states from the j-transition prob
			if(no_s0 || no_s1)
			{
				outf << ">> Missing states in S from T("+s0[j]+","+A[i]+","+s1[j]+": ";
				if(no_s0) outf << s0[j] + " ";
				if(no_s1) outf << s1[j] + " ";
				outf << endl;
			}
		}
	}

	//Check if all states & observations from the O-function are in S & Z
	for(unsigned int i = 0; i < O.size(); i++)
	{
		vector<string>::const_iterator ite;
		vector<string> s0 = get<0>(O[i]);
		vector<string> ob = get<1>(O[i]);
		for(unsigned int j = 0; j < s0.size(); j++)
		{
			bool no_s0(false);
			bool no_ob(false);
			ite = find(S.begin(),S.end(),s0[j]);
			if(ite == S.end()) no_s0 = true;
			ite = find(Z.begin(),Z.end(),ob[j]);
			if(ite == Z.end()) no_ob = true;

			//Display missing elements from the j-observation prob
			if(no_s0 || no_ob)
			{
				outf << ">> Missing elements in S & Z from O("+s0[j]+","+A[i]+","+ob[j]+": ";
				if(no_s0) outf << s0[j] + " ";
				if(no_ob) outf << ob[j] + " ";
				outf << endl;
			}
		}
	}

	//Check that every observation from Z has at least one non-zero probabiliy of being perceived
	for(unsigned int i = 0; i < Z.size(); i++)
	{
		vector<string>::const_iterator ite;
		bool can_be_perc(false);
		for(unsigned int j = 0; j < O.size(); j++)
		{
			vector<string> ob = get<1>(O[j]);
			ite = find(ob.begin(),ob.end(),Z[i]);
			if(ite != ob.end())
			{
				can_be_perc = true;
				break;
			}
		}
		if(!can_be_perc)
		{
			outf << ">> Observation (" + Z[i] + ") does not appear in any non-zero O-probability." << endl;
		}

	}

	outf << "If this file contains no lines that start with \">>\", then the checked POMDP is good to go :)" << endl;
	outf.close();

	//Check from the final diagnosis file if this POMDP is OK
	ifstream infile("pomdp-debug.txt");
	bool is_ok(true);
	string tmp;
	while(getline(infile,tmp))
	{
		if(tmp.length() > 0)
		{
			if(tmp[0] == '>')
			{
				is_ok = false;
				break;
			}
		}
	}
	infile.close();

	return is_ok;
}

pomdp HPomdp::generatePomdp(vector<string> const &S,
			      vector<string> const &A,
			      vector<string> const &Z,
			      vector<pMat> const &T,
			      vector<pMat> const &O,
			      vector<string> const &G,
			      vector<tuple<string,string> > const &Gpair,
			      vector<string> const &P,
			      vector<tuple<string,string> > const &Ppair,
			      double const &discount,
			      vector<string> const &rewOrder,
			      bool &succ)
{
	//Output model
	POMDP::Model<MDP::Model> model(1,1,1);
	vector<string> tsv;
	vector<pMat> tpmv;
	vector<tuple<string,string> > ttsv;

	//Initial check for input parameters
	succ = true;
	if(discount < 0 || discount > 1){ succ = false; cout << "1";}
	if(S.size() == 0){ succ = false; cout << "2";}
	if(A.size() == 0){ succ = false; cout << "3";}
	if(Z.size() == 0){ succ = false; cout << "4";}
	if(T.size() == 0){ succ = false; cout << "5";}
	if(O.size() == 0){ succ = false; cout << "6";}
	if(A.size() != O.size()){ succ = false; cout << "7";}
	if(A.size() != T.size()){ succ = false; cout << "8";}
	if(!succ)
	{
		cout << ">> generatePomdp[error]: Failed initial check." << endl;
		return pomdp(model,tsv,tsv,tsv,tpmv,tpmv,tsv,ttsv,tsv,ttsv,0);
	}


	//Create a Cassandra file
	shellCmd(string("rm ") + string("temporalPomdpFile.POMDP"));
	ofstream ofile;
	ofile.open("temporalPomdpFile.POMDP");

	//Write the discount, value, states, actions & observations
	ofile << "discount: " << discount << endl;
	ofile << "values: reward" << endl;
	ofile << "states: ";
	for(unsigned int i = 0; i < S.size(); i++) ofile << S[i] << " ";
	ofile << endl;
	ofile << "actions: ";
	for(unsigned int i = 0; i < A.size(); i++) ofile << A[i] << " ";
	ofile << endl;
	ofile << "observations: ";
	for(unsigned int i = 0; i < Z.size(); i++) ofile << Z[i] << " ";
	ofile << endl << endl;

	//Write the transition function
	for(unsigned int i = 0; i < T.size(); i++)
	{
		//Get vectors of starting state, ending state & transition probability
		vector<string> s0 = get<0>(T[i]);
		vector<string> s1 = get<1>(T[i]);
		vector<double> t = get<2>(T[i]);

		//Iterate over every 'starting state' (rows of the i-th matrix)
		ofile << "T: " + A[i] << endl;
		for(unsigned int j = 0; j < S.size(); j++)
		{
			//Gather the transitions for the i-th action in which the 
			//j-th state is the 'starting state'
			vector<unsigned> row_t;
			for(unsigned int k = 0; k < s0.size(); k++)
			{
				if(s0[k] == S[j]) row_t.push_back(k);
			}

			//Verify that there is at least one transition
			//that starts in the j-th state
			if(row_t.size() == 0)
			{
				cout << ">> generatePomdp[error]: No transtion starts in " + S[j] + " with action " + A[i] << endl;
				succ = false;
				return pomdp(model,tsv,tsv,tsv,tpmv,tpmv,tsv,ttsv,tsv,ttsv,0);
			}

			//Write the j-th row of the i-th transition matrix (action)
			bool atleast_end_state(false);
			for(unsigned int k = 0; k < S.size(); k++)
			{
				bool zeroprob_t = true;
				for(unsigned int l = 0; l < row_t.size(); l++)
				{
					if(S[k] == s1[row_t[l]])
					{
						atleast_end_state = true;
						zeroprob_t = false;
						ofile << t[row_t[l]] << " ";
						break;
					}
				}

				//The prob. of transit from the j-th to the
				//k-th state with the i-th action is 0
				if(zeroprob_t) ofile << "0 ";
			}
			ofile << endl;

			//Veirfy that at least one transition starts
			//in j-th state & ends in a state of 'S'
			if(!atleast_end_state)
			{
				cout << ">> generatePomdp[error]: No transtion starts in " + S[j] + " with action " + A[i] + " and ends in a state of S." << endl;
				succ = false;
				return pomdp(model,tsv,tsv,tsv,tpmv,tpmv,tsv,ttsv,tsv,ttsv,0);
			}
		}

		//Add an empty line between each transition matrix
		ofile << endl;
	}

	//Write the observation function
	for(unsigned int i = 0; i < O.size(); i++)
	{
		//Get vectors of starting state, perceived observation & observation probability
		vector<string> s0 = get<0>(O[i]);
		vector<string> zz = get<1>(O[i]);
		vector<double> o = get<2>(O[i]);

		//Iterate over every 'starting state' (rows of the i-th matrix)
		ofile << "O: " + A[i] << endl;
		for(unsigned int j = 0; j < S.size(); j++)
		{
			//Gather the observations for the i-th action in which the 
			//j-th state is the 'starting state'
			vector<unsigned> row_o;
			for(unsigned int k = 0; k < s0.size(); k++)
			{
				if(s0[k] == S[j]) row_o.push_back(k);
			}

			//Verify that there is at least one transition
			//that starts in the j-th state
			if(row_o.size() == 0)
			{
				cout << ">> generatePomdp[error]: There is no obs. prob. for state " + S[j] + " with action " + A[i] << endl;
				succ = false;
				return pomdp(model,tsv,tsv,tsv,tpmv,tpmv,tsv,ttsv,tsv,ttsv,0);
			}

			//Write the j-th row of the i-th observation matrix (action)
			bool atleast_obs(false);
			for(unsigned int k = 0; k < Z.size(); k++)
			{
				bool zeroprob_o = true;
				for(unsigned int l = 0; l < row_o.size(); l++)
				{
					if(Z[k] == zz[row_o[l]])
					{
						atleast_obs = true;
						zeroprob_o = false;
						ofile << o[row_o[l]] << " ";
						break;
					}
				}

				//The prob. of transit from the j-th to the
				//k-th state with the i-th action is 0
				if(zeroprob_o) ofile << "0 ";
			}
			ofile << endl;

			//Veirfy that at least one transition starts
			//in j-th state & ends in a state of 'S'
			if(!atleast_obs)
			{
				cout << ">> generatePomdp[error]: There is no obs. prob. for state " + S[j] + " with action " + A[i] + " and an obs. of Z." << endl;
				succ = false;
				return pomdp(model,tsv,tsv,tsv,tpmv,tpmv,tsv,ttsv,tsv,ttsv,0);
			}
		}

		//Add an empty line between each observation matrix
		ofile << endl;
	}

	//Write the reward function
	//First make  sure  that all the goal & punishment states are in the 'S' vector
	vector<string>::const_iterator ite;
	vector<string> tmp,tmp_;
	for(unsigned int i = 0; i < G.size(); i++)
	{
		if(G[i] == "*") continue;

		ite = find(S.begin(),S.end(),G[i]);
		if(ite == S.end()) tmp.push_back(G[i]);
	}
	for(unsigned int i = 0; i < P.size(); i++)
	{
		if(P[i] == "*") continue;

		ite = find(S.begin(),S.end(),P[i]);
		if(ite == S.end()) tmp_.push_back(P[i]);
	}
	if(tmp.size() > 0 || tmp_.size() > 0)
	{
		cout << ">> generatePomdp[error]: Non-existing goal states[";
		for(unsigned int i = 0; i < tmp.size(); i++) cout << tmp[i] + ",";
		cout << "]. Non-existing punishment states[";
		for(unsigned int i = 0; i < tmp_.size(); i++) cout << tmp_[i] + ",";
		cout << "]." << endl;

		succ = false;
		return pomdp(model,tsv,tsv,tsv,tpmv,tpmv,tsv,ttsv,tsv,ttsv,0);
	}

	//Check that every state & action in the goal & punishment pairs exist in 'S' & 'A'
	tmp.clear();
	tmp_.clear();
	for(unsigned int i = 0; i < Gpair.size(); i++)
	{
		string s_ = get<0>(Gpair[i]);
		string a_ = get<1>(Gpair[i]);

		if(s_ != "*")
		{
			ite = find(S.begin(),S.end(),s_);
			if(ite == S.end()) tmp.push_back(s_);
		}

		if(a_ != "*")
		{
			ite = find(A.begin(),A.end(),a_);
			if(ite == A.end()) tmp_.push_back(a_);
		}
	}
	if(tmp.size() > 0 || tmp_.size() > 0)
	{
		cout << ">> generatePomdp[error]: Non-existing goal states[";
		for(unsigned int i = 0; i < tmp.size(); i++) cout << tmp[i] + ",";
		cout << "]. Non-existing goal actions[";
		for(unsigned int i = 0; i < tmp_.size(); i++) cout << tmp_[i] + ",";
		cout << "]." << endl;

		succ = false;
		return pomdp(model,tsv,tsv,tsv,tpmv,tpmv,tsv,ttsv,tsv,ttsv,0);
	}
	tmp.clear();
	tmp_.clear();
	for(unsigned int i = 0; i < Ppair.size(); i++)
	{
		string s_ = get<0>(Ppair[i]);
		string a_ = get<1>(Ppair[i]);

		if(s_ != "*")
		{
			ite = find(S.begin(),S.end(),s_);
			if(ite == S.end()) tmp.push_back(s_);
		}

		if(a_ != "*")
		{
			ite = find(A.begin(),A.end(),a_);
			if(ite == A.end()) tmp_.push_back(a_);
		}
	}
	if(tmp.size() > 0 || tmp_.size() > 0)
	{
		cout << ">> generatePomdp[error]: Non-existing punishment states[";
		for(unsigned int i = 0; i < tmp.size(); i++) cout << tmp[i] + ",";
		cout << "]. Non-existing punishment actions[";
		for(unsigned int i = 0; i < tmp_.size(); i++) cout << tmp_[i] + ",";
		cout << "]." << endl;

		succ = false;
		return pomdp(model,tsv,tsv,tsv,tpmv,tpmv,tsv,ttsv,tsv,ttsv,0);
	}

	//Write the default reward for every action
	for(unsigned int i = 0; i < A.size(); i++)
	{
		ofile << "R: " + A[i] + " : * : * : * -1" << endl << endl;
	}

	//Check that the reward printing order  vector is a valid one
	vector<string> order;
	bool use_def(false);
	if(rewOrder.size() != 4) use_def = true;
	else
	{
		bool g(false);
		bool p(false);
		bool gp(false);
		bool pp(false);
		for(unsigned int i = 0; i < rewOrder.size(); i++)
		{
			if(rewOrder[i] == "g") g = true;
			if(rewOrder[i] == "p") p = true;
			if(rewOrder[i] == "gp") gp = true;
			if(rewOrder[i] == "pp") pp = true;
		}

		use_def = !(g && p && gp && pp);
	}

	//Set the reward printing order
	if(use_def)
	{
		order.push_back("g");
		order.push_back("p");
		order.push_back("gp");
		order.push_back("pp");
	}
	else order = rewOrder;

	//Print the reward function using the set order
	//Overwrite the reward goal & punishment states & state-action pairs
	for(unsigned int i = 0; i < order.size(); i++)
	{
		if(order[i] == "g")
		{
			for(unsigned int j = 0; j < G.size(); j++)
			{
				ofile << "R: * : * : " + G[j] + " : * 100" << endl << endl;
			}
		}
		else if(order[i] == "p")
		{
			for(unsigned int j = 0; j < P.size(); j++)
			{
				ofile << "R: * : * : " + P[j] + " : * -100" << endl << endl;
			}
		}
		else if(order[i] == "gp")
		{
			for(unsigned int j = 0; j < Gpair.size(); j++)
			{
				string s_ = get<0>(Gpair[j]);
				string a_ = get<1>(Gpair[j]);
				ofile << "R: " + a_ + " : " + s_ + " : * : * 100" << endl << endl;
			}
		}
		else if(order[i] == "pp")
		{
			for(unsigned int j = 0; j < Ppair.size(); j++)
			{
				string s_ = get<0>(Ppair[j]);
				string a_ = get<1>(Ppair[j]);
				ofile << "R: " + a_ + " : " + s_ + " : * : * -100" << endl << endl;
			}
		}
	}

	//Close POMDP file so it can be read
	ofile.close();
	//Read the Cassandra file to generate the POMDP::Model<MDP::Model>
	ifstream infile;
	infile.open("temporalPomdpFile.POMDP");
	Impl::CassandraParser cp;
	auto p_vals = cp.parsePOMDP(infile);
	size_t num_S = get<0>(p_vals);
	size_t num_A = get<1>(p_vals);
	size_t num_O = get<2>(p_vals);
	auto TT = get<3>(p_vals);
	auto RR = get<4>(p_vals);
	auto OO = get<5>(p_vals);
	double discount_ = get<6>(p_vals);

	//Perform a truncate step on the T and O distributions to make sure echa one adds to 1
	//NOTE: Precision errors might arise after reading the distributions from the Cassandr file
	unsigned trunc_prec(5);
	for(unsigned int i = 0; i < num_A; i++)
	{
		for(unsigned int j = 0; j < TT.size(); j++)
		{
			double sum(0.0);
			double max(-1.0);
			int id(-1);
			for(unsigned int k = 0; k < TT[j][i].size(); k++)
			{
				TT[j][i][k] = truncate(TT[j][i][k],trunc_prec);
				sum += TT[j][i][k];
				if(TT[j][i][k] > max)
				{
					max = TT[j][i][k];
					id = k;
				}
			}

			//Compensate the total-sum error on the element  with highest probability
			double diff = static_cast<double>(1.0) - sum;
			if(diff > 0.0) TT[j][i][id] += diff;
			else if(diff < 0.0) TT[j][i][id] -= diff;
		}
	}
	for(unsigned int i = 0; i < num_A; i++)
	{
		for(unsigned int j = 0; j < OO.size(); j++)
		{
			double sum(0.0);
			double max(-1.0);
			int id(-1);
			for(unsigned int k = 0; k < OO[j][i].size(); k++)
			{
				OO[j][i][k] = truncate(OO[j][i][k],trunc_prec);
				sum += OO[j][i][k];
				if(OO[j][i][k] > max)
				{
					max = OO[j][i][k];
					id = k;
				}
			}

			//Compensate the total-sum error on the element  with highest probability
			double diff = static_cast<double>(1.0) - sum;
			if(diff > 0.0) OO[j][i][id] += diff;
			else if(diff < 0.0) OO[j][i][id] -= diff;
		}
	}

////debug
//cout << std::fixed << std::setprecision(15);
//int tt(0);
//int oo(0);
//for(unsigned int i = 0; i < num_A; i++)
//{
//	for(unsigned int j = 0; j < TT.size(); j++)
//	{
//		double sum(0.0);
//		for(unsigned int k = 0; k < TT[j][i].size(); k++) sum += TT[j][i][k];
//		if(sum != 1.0)
//		{
//			cout << ">> T: "+S[j]+" x "+A[i]+": " << sum << endl;
//			tt++;
//		}
//	}
//}
//for(unsigned int i = 0; i < num_A; i++)
//{
//	for(unsigned int j = 0; j < OO.size(); j++)
//	{
//		double sum(0.0);
//		for(unsigned int k = 0; k < OO[j][i].size(); k++) sum += OO[j][i][k];
//		if(sum != 1.0)
//		{
//			cout << ">> O: "+S[j]+" x "+A[i]+": " << sum << endl;
//			oo++;
//		}
//	}
//}
//cout << "wrong T distributions: " << tt << endl;
//cout << "wrong O distributions: " << oo << endl;

	POMDP::Model<MDP::Model> output_model(num_O,num_S,num_A);
	output_model.setTransitionFunction(TT);
	output_model.setRewardFunction(RR);
	output_model.setObservationFunction(OO);
	output_model.setDiscount(discount_);

////debug
//cout << "*********************" << endl;
//cout << "------ Inside generatePomdp ------" << endl;
//cout << "\t#S: " << num_S << endl;
//cout << "\t#A: " << num_A << endl;
//cout << "\t#Z: " << num_O << endl;
//cout << "\tT-dims: " << TT.size() << "x" << TT[0].size() << "x" << TT[0][0].size() << endl;
//cout << "\tO-dims: " << OO.size() << "x" << OO[0].size() << "x" << OO[0][0].size() << endl;
//cout << "\tR-dims: " << RR.size() << "x" << RR[0].size() << "x" << RR[0][0].size() << endl;
//cout << "TRANSITION MATRICES" << endl;
//for(unsigned int i = 0; i < TT.size(); i++)
//{
//	cout << endl;
//	for(unsigned int j = 0; j < TT[i].size(); j++)
//	{
//		for(unsigned int k = 0; k < TT[i][j].size(); k++)
//		{
//			cout << TT[i][j][k] << " ";
//		}
//		cout << endl;
//	}
//}
//cout << "*********************" << endl;

	//Close POMDP file
	infile.close();

	//Return the POMDP with the list of states, actions & observations as a 'pomdp' typedef
	unsigned horizon = 0;
	return pomdp(output_model,S,A,Z,T,O,G,Gpair,P,Ppair,horizon);
}

void HPomdp::generateBottomModel()
{
	if(!env_loaded_) return;

	//Clear the POMDP vectors
	S_.clear();
	A_.clear();
	Z_.clear();
	T_.clear();
	O_.clear();
	AA_.clear();

	//Get all the states at the bottom of the hierarchy
	unsigned depth = hs_.depth();
	vector<string> con_S = hs_.keysAtLevel(depth - 1);

	//Set of concrete states
	S_.push_back(con_S);

	//Set of concrete actions
	A_.push_back(concrete_A);

	//Add the concrete actions to the vector of abstract actions (necessary to build the tree of abstract actions)
	for(unsigned int i = 0; i < concrete_A.size(); i++)
	{
		vector<string> tmp_ao;
		POMDP::Policy tmp_pol(1,1,1);
		POMDP::Model<MDP::Model> tmp_m(1,1,1,0.9);
		policy pol(tmp_pol,tmp_ao,tmp_ao,tmp_ao,tmp_ao,tmp_m);
		AA_.insert(pair<string,policy>(concrete_A[i],pol));
	}

	//Determine if the the O-function will be built using the default prob. or the distribution
	bool use_dist = (o_dist.size() > 0);

	//Set of concrete observations
	//if(!use_dist) con_S.push_back("none");
	Z_.push_back(con_S);

	//Transition & Observation probability functions
	vector<pMat> bottom_T;
	vector<pMat> bottom_O;
	for(unsigned int i = 0; i < A_[0].size(); i++)
	{
		//T function params: starting-state. ending-state & transition-prob
		vector<string> s0,s1;
		vector<double> tp;

		//O function params: current-state, observation & observation-prob
		vector<string> cs, ob;
		vector<double> op;

		//Iterate over each state at the bottom of the hierarchy
		for(unsigned int j = 0; j < S_[0].size(); j++)
		{
			vector<string> neig;
			bool result;

			if(A_[0][i] == "up") neig = nh_.aboveOf(S_[0][j],result);
			else if(A_[0][i] == "down") neig = nh_.belowOf(S_[0][j],result);
			else if(A_[0][i] == "left") neig = nh_.leftOf(S_[0][j],result);
			else if(A_[0][i] == "right") neig = nh_.rightOf(S_[0][j],result);

			//T
			if(neig.size() > 0)
			{
				//Transit to the ending state
				s0.push_back(S_[0][j]);
				s1.push_back(neig[0]);
				tp.push_back(concrete_T[i]);
				s0.push_back(S_[0][j]);
				s1.push_back(S_[0][j]);
				tp.push_back(1 - concrete_T[i]);
			}
			else
			{
				//Stay in the starting state
				s0.push_back(S_[0][j]);
				s1.push_back(S_[0][j]);
				tp.push_back(1);
			}

			//O
			if(!use_dist)
			{
				//Use the observation precision
				cs.push_back(S_[0][j]);
				ob.push_back(S_[0][j]);
				op.push_back(concrete_O[0]);//Precision of perceiving the observation
				cs.push_back(S_[0][j]);
				ob.push_back("none");
				op.push_back(1 - concrete_O[0]);//Precision of perceiving the observation
			}
			if(use_dist)
			{
				//Use Gaussian probability distribution

				//Set the ending cell as center of the kernel
				string center_cell = S_[0][j];

				vector<string> cs_tmp, ob_tmp;
				vector<double> op_tmp;
				for(unsigned int k = 0; k < o_dist.size(); k++)
				{
					int x = get<0>(o_dist[k]);
					int y = get<1>(o_dist[k]);
					double p = get<2>(o_dist[k]);

					//Its the prob. of the central cell
					if(x == 0 && y == 0)
					{
						cs_tmp.push_back(S_[0][j]);
						ob_tmp.push_back(center_cell);
						op_tmp.push_back(p);
					}
					//A neighbor cell whose connectivity to the central cell 
					//must be checked before adding it as a possible observation
					else
					{
						int abs_x = abs(x);
						int abs_y = abs(y);
						string x_most,y_most,tgt_cell;

						vector<string> v_tmp;
						string curr_c(center_cell);
						bool cell_found(false);
						for(int l = 0; l < abs_x; l++)
						{
							if(x > 0) v_tmp = nh_.rightOf(curr_c,cell_found);
							else v_tmp = nh_.leftOf(curr_c,cell_found);

							if(!cell_found) break;
							else curr_c = v_tmp[0];
						}
						x_most = curr_c;

						//Check that the furthest X was reached
						if(!cell_found && abs_x > 0) continue;

						curr_c = center_cell;
						for(int l = 0; l < abs_y; l++)
						{
							if(y > 0) v_tmp = nh_.aboveOf(curr_c,cell_found);
							else v_tmp = nh_.belowOf(curr_c,cell_found);

							if(!cell_found) break;
							else curr_c = v_tmp[0];
						}
						y_most = curr_c;

						//Check that the furthest Y was reached
						if(!cell_found) continue;

						//Now check if the target-cell can be reached from both the  X & Y furthest
						//From furth-Y to target
						curr_c = y_most;
						for(int l = 0; l < abs_x; l++)
						{
							if(x > 0) v_tmp = nh_.rightOf(curr_c,cell_found);
							else v_tmp = nh_.leftOf(curr_c,cell_found);

							if(!cell_found) break;
							else curr_c = v_tmp[0];
						}

						//Check if target cell reached from the furthest Y
						if(!cell_found) continue;

						//From furth-X to target
						curr_c = x_most;
						for(int l = 0; l < abs_y; l++)
						{
							if(y > 0) v_tmp = nh_.aboveOf(curr_c,cell_found);
							else v_tmp = nh_.belowOf(curr_c,cell_found);

							if(!cell_found) break;
							else curr_c = v_tmp[0];
						}

						//The target cell can be reached from both paths,
						//& therefore it cen be perceived
						if(cell_found)
						{
							cs_tmp.push_back(S_[0][j]);
							ob_tmp.push_back(curr_c);
							op_tmp.push_back(p);

						}
					}
				}

				//Normalize the probabilitiy for the valid observations for the j-th state
				double total_p(0.0);
				double real_total_p(0.0);
				for(unsigned int k = 0; k < op_tmp.size(); k++) total_p += op_tmp[k];
				for(unsigned int k = 0; k < op_tmp.size(); k++)
				{
					op_tmp[k] /= total_p;
					real_total_p += op_tmp[k];
				}
				for(unsigned int k = 0; k < ob_tmp.size(); k++)
				{
					//Make sure the total probabilities sum 1
					if(ob_tmp[k] == cs_tmp[k])
					{
						//Add the remaining to the central observation
						op_tmp[k] += (1 - real_total_p);
						break;
					}
				}

				//Append the observation probabilities
				cs.insert(cs.end(),cs_tmp.begin(),cs_tmp.end());
				ob.insert(ob.end(),ob_tmp.begin(),ob_tmp.end());
				op.insert(op.end(),op_tmp.begin(),op_tmp.end());
			}
		}

		//Build i-th action's T & O function
		pMat a_T(s0,s1,tp);
		pMat a_O(cs,ob,op);

		//Save i-th action's functions
		bottom_T.push_back(a_T);
		bottom_O.push_back(a_O);
	}

	//Transition & Observation functions for concrete components
	T_.push_back(bottom_T);
	O_.push_back(bottom_O);

	//Create the AIToolbox model for the bottom level POMDP
	vector<string> gp,order;
	vector<tuple<string,string> > gpp;
	order.push_back("g");
	order.push_back("p");
	order.push_back("pp");
	order.push_back("gp");
	bool success;
	auto p = generatePomdp(S_[0],A_[0],Z_[0],T_[0],O_[0],gp,gpp,gp,gpp,0.95,order,success);
	M_.clear();
	M_.push_back(get<0>(p));
}

void HPomdp::generateBottomZSMap()
{
	//Set of observations at bottom model
	vector<string> set_Z = Z_[0];

	//O function at bottom model
	vector<pMat> fun_O = O_[0];

	//Iterate over set_Z
	vector<string>::iterator ite;
	for(unsigned int i = 0; i < set_Z.size(); i++)
	{
		vector<string> tmp_vec;

		//Iterarete over actions
		for(unsigned int j = 0; j < fun_O.size(); j++)
		{
			//Amount of O-tuples at bottom model in the j-th action
			unsigned n_ot = std::get<0>(fun_O[j]).size();

			//Iterate over O-tuples
			for(unsigned int  k = 0; k < n_ot; k++)
			{
				if(std::get<1>(fun_O[j])[k] == set_Z[i])
				{
					string tmp_s = std::get<0>(fun_O[j])[k];
					ite = std::find(tmp_vec.begin(),tmp_vec.end(),tmp_s);
					if(ite == tmp_vec.end()) tmp_vec.push_back(tmp_s);
				}
			}
		}

		//Add the mapping from the i-th observation to its states
		z2s_[set_Z[i]] = tmp_vec;
	}
}

void HPomdp::buildHierarchyAA(int const &n_sim_aa_params)
{
	if(!env_loaded_) return;

	//Get the ply at the hierarchical state space bottom level
	unsigned bottom_level = hs_.depth() - 1;

	//Go from the bottom of hs_ to the top
	for(unsigned int i = bottom_level; i > 0; i--)
	{
		//check if the level above the current one has 
		//more than one abstract state
		vector<string> abs_s = hs_.keysAtLevel(i-1);
		if(abs_s.size() == 1)
		{
			if(abs_s[0] == "root") break;
			else
			{
				vector<string> S_tmp;
				vector<string> A_tmp;
				vector<string> Z_tmp;
				vector<pMat> T_tmp;
				vector<pMat> O_tmp;

				//Add the only state to the global vector and the other empty vectors
				if(abs_s.size() == 1) S_tmp.push_back(abs_s[0]);
				S_.push_back(S_tmp);
				A_.push_back(A_tmp);
				Z_.push_back(Z_tmp);
				T_.push_back(T_tmp);
				O_.push_back(O_tmp);

				continue;
			}
		} 
		else if(abs_s.size() == 0) break;

		//info
		cout << ">> Processing "<< abs_s.size() <<" states at level " << i << ":" << endl;

		//Parameters for the model of the level immediately above the i-th level
		vector<string> S_tmp;
		vector<string> A_tmp;
		vector<string> Z_tmp;
		vector<pMat> T_tmp;
		vector<pMat> O_tmp;

		//Create abstract actions that start in the j-th abstract state
		for(unsigned int j = 0; j < abs_s.size(); j++)
		{
			//info
			if(j > 0) cout << ",";
			cout << j+1;
			cout.flush();
			//cout << "\t>> Processing state " << j+1 << "/" << abs_s.size() << endl;

			//Get the j-th abstract state's neighbor states
			bool res;
			vector<string> neig_s = nh_.neigTo(abs_s[j],res);

			//Get the j-th abs. state children states
			vector<string> j_chi = hs_.keysOfChildren(abs_s[j]);

			//Build vector of all peripheral states to the j-th abstract state
			vector<string> peri_s;
			for(unsigned int k = 0; k < j_chi.size(); k++)
			{
				//Get neighbors that are not children of the j-th abs. state
				vector<string> tmp = nh_.neigToExc(j_chi[k],j_chi);

				//Make to not have repeated peripheral states
				vector<string>::iterator ite;
				for(unsigned int l = 0; l < tmp.size(); l++)
				{
					ite = find(peri_s.begin(),peri_s.end(),tmp[l]);
					if(ite == peri_s.end()) peri_s.push_back(tmp[l]);
				}

			}

			//Get the peripheral goal states to transit to each neighbor of the j-th abstract state
			//Also, build a vector with the parent of every peripheral state
			vector<vector<string> > goal_s;
			vector<string> parent_peri_s(peri_s.size(),string(""));
			for(unsigned int k = 0; k < neig_s.size(); k++)
			{
				vector<string> tmp;
				vector<string> n_chi = hs_.keysOfChildren(neig_s[k]);
				vector<string>::iterator ite;
				for(unsigned int l = 0; l < n_chi.size(); l++)
				{
					ite = find(peri_s.begin(),peri_s.end(),n_chi[l]);
					if(ite != peri_s.end())
					{
						tmp.push_back(n_chi[l]);

						int idx = distance(peri_s.begin(),ite);
						parent_peri_s[idx] = neig_s[k];
					}
				}
				goal_s.push_back(tmp);
			}

			//Build the POMDP parameters
			vector<string> S = j_chi;
			vector<string> A;
			vector<string> Z;
			vector<pMat> T;
			vector<pMat> O;

			//Add the goal peripheral states
			S.insert(S.end(),peri_s.begin(),peri_s.end());

			//Add the extra state that models region of space not considered in this POMDP
			S.push_back("extra");

			//Add actions that have at least one non-zero-prob transition that starts in a
			//state contained in this POMDP's S set and that transits to a different state
			unsigned top_lvl = T_.size() - 1;
			vector<string>::iterator ite;
			for(unsigned int k = 0; k < T_[top_lvl].size(); k++)
			{
				bool add_action(false);
				vector<string> s0 = get<0>(T_[top_lvl][k]);
				vector<string> s1 = get<1>(T_[top_lvl][k]);
				for(unsigned int l = 0; l < S.size(); l++)
				{
					for(unsigned int m = 0; m < s0.size(); m++)
					{
						if(S[l] == s0[m] && S[l] != s1[m]) add_action = true;

						if(add_action) break;
					}

					if(add_action) break;
				}

				//Add the k-the action
				if(add_action) A.push_back(A_[top_lvl][k]);
			}

			//Add transitions that start in a state of 'S'
			for(unsigned int k = 0; k < A.size(); k++)
			{
				vector<string> tmp_s0;
				vector<string> tmp_s1;
				vector<double> tmp_tp;

				ite = find(A_[top_lvl].begin(),A_[top_lvl].end(),A[k]);
				int idx = distance(A_[top_lvl].begin(),ite);
				vector<string> s0 = get<0>(T_[top_lvl][idx]);
				vector<string> s1 = get<1>(T_[top_lvl][idx]);
				vector<double> tp = get<2>(T_[top_lvl][idx]);

				for(unsigned int l = 0; l < s0.size(); l++)
				{
					ite = find(S.begin(),S.end(),s0[l]);
					if(ite != S.end())
					{
						//Add the l-th transition
						tmp_s0.push_back(s0[l]);
						tmp_tp.push_back(tp[l]);

						//Check if the l-th transition ends in S or in the 'extra' state
						ite = find(S.begin(),S.end(),s1[l]);
						if(ite != S.end()) tmp_s1.push_back(s1[l]);
						else tmp_s1.push_back("extra");
					}
				}

				//Sum those transitions' prob. that start in the same state & end in the 'extra' state
				while(true)
				{
					int ori(-1);
					vector<int> rep;
					for(unsigned int l = 0; l < tmp_s1.size(); l++)
					{
						if(tmp_s1[l] == "extra")
						{
							for(unsigned int m = l+1; m < tmp_s0.size(); m++)
							{
								if(tmp_s0[l] == tmp_s0[m] && tmp_s1[m] == "extra")
								{
									ori = l;
									rep.push_back(m);
								}
							}
						}

						if(ori >= 0) break;
					}

					//Cluster the repreated transitions
					if(ori >= 0)
					{
						for(int l = rep.size()-1; l >= 0; l--)
						{
							//Sum the probability
							tmp_tp[ori] += tmp_tp[rep[l]];

							//Delete the repeated transition
							tmp_s0.erase(tmp_s0.begin()+rep[l]);
							tmp_s1.erase(tmp_s1.begin()+rep[l]);
							tmp_tp.erase(tmp_tp.begin()+rep[l]);
						}
					}

					//No repeated transitions were found in this cycle
					if(ori == -1) break;
				}

				//Add the T(s,a) distribution for the 'extra' state and the k-th action
				tmp_s0.push_back("extra");
				tmp_s1.push_back("extra");
				tmp_tp.push_back(1.0);

				//Add k-th action's T distribution
				pMat a_T(tmp_s0,tmp_s1,tmp_tp);
				T.push_back(a_T);
			}

			//Add the extra observation that models those observations not considered in this POMDP
			Z.push_back("extra");

			//Add observations if their prob. of being perceived when reaching a state in 'S' is non-zero
			//Also add their observation distributions
			for(unsigned int k = 0; k < A.size(); k++)
			{
				ite = find(A_[top_lvl].begin(),A_[top_lvl].end(),A[k]);
				int idx = distance(A_[top_lvl].begin(),ite);
				vector<string> ss = get<0>(O_[top_lvl][idx]);
				vector<string> ob = get<1>(O_[top_lvl][idx]);
				vector<double> op = get<2>(O_[top_lvl][idx]);

				vector<string> tmp_ss;
				vector<string> tmp_ob;
				vector<double> tmp_op;
				for(unsigned int l = 0; l < ss.size(); l++)
				{
					ite = find(S.begin(),S.end(),ss[l]);
					if(ite != S.end())
					{
						ite = find(Z.begin(),Z.end(),ob[l]);
						if(ite == Z.end()) Z.push_back(ob[l]);

						//Add the O(s,z,a) transition
						tmp_ss.push_back(ss[l]);
						tmp_ob.push_back(ob[l]);
						tmp_op.push_back(op[l]);
					}
				}

				//Add the O(s,a) distribution for the 'extra' state and the k-th action
				tmp_ss.push_back("extra");
				tmp_ob.push_back("extra");
				tmp_op.push_back(1.0);

				//Add k-th action's O distribution
				pMat a_O(tmp_ss,tmp_ob,tmp_op);
				O.push_back(a_O);
			}

			//Check if the absb-goal, absb-nongoal, terminate & none components have not been added, if so add them
			ite = find(S.begin(),S.end(),"absb-g");
			if(ite == S.end()) S.push_back("absb-g");
			ite = find(S.begin(),S.end(),"absb-ng");
			if(ite == S.end()) S.push_back("absb-ng");
			ite = find(Z.begin(),Z.end(),"none");
			if(ite == Z.end()) Z.push_back("none");
			ite = find(A.begin(),A.end(),"terminate");
			if(ite == A.end())
			{
				//Add the terminate action
				A.push_back("terminate");

				//Add empty T & O matrices
				vector<string> tmp1,tmp2;
				vector<double> tmp3;
				T.push_back(pMat(tmp1,tmp2,tmp3));
				O.push_back(pMat(tmp1,tmp2,tmp3));
			}

			//Add the T & O probabilities for actions executed from the absorbent states
			for(unsigned int k = 0; k < A.size(); k++)
			{
				//k-th action T-prob
				get<0>(T[k]).push_back("absb-g");
				get<1>(T[k]).push_back("absb-g");
				get<2>(T[k]).push_back(1.0);
				get<0>(T[k]).push_back("absb-ng");
				get<1>(T[k]).push_back("absb-ng");
				get<2>(T[k]).push_back(1.0);

				//k-th action O-prob
				get<0>(O[k]).push_back("absb-g");
				get<1>(O[k]).push_back("none");
				get<2>(O[k]).push_back(1.0);
				get<0>(O[k]).push_back("absb-ng");
				get<1>(O[k]).push_back("none");
				get<2>(O[k]).push_back(1.0);
			}

			//Add the T & O probabilities for executing "terminate" from 
			//every state other than "absb-g" & "absb-ng"
			ite = find(A.begin(),A.end(),"terminate");
			int term_index = distance(A.begin(),ite);
			for(unsigned int k = 0; k < S.size(); k++)
			{
				if(S[k] == "absb-g" || S[k] == "absb-ng") continue;

				//NOTE: temporary, every state will transit to "absb-ng" after "terminate", however,
				//some of these transitions will be modified to end in "absb-g" when the POMDP's
				//goal states are defined.

				//(k-th state, terminate) T-prob
				get<0>(T[term_index]).push_back(S[k]);
				get<1>(T[term_index]).push_back("absb-ng");
				get<2>(T[term_index]).push_back(1.0);

				//(k-th state, terminate) O-prob
				get<0>(O[term_index]).push_back(S[k]);
				get<1>(O[term_index]).push_back("none");
				get<2>(O[term_index]).push_back(1.0);
			}

			//Check that the T & O distribution for every (state-action) pair sums 1
			for(unsigned int k = 0; k < A.size(); k++)
			{
				//Get the T & O functions for the k-th action
				vector<string> s0 = get<0>(T[k]);
				vector<double> tp = get<2>(T[k]);
				vector<string> ss = get<0>(O[k]);
				vector<double> op = get<2>(O[k]);

				//Iterate over every state
				for(unsigned int l = 0; l < S.size(); l++)
				{
					//Truncate precision for the T-function
					double acc_p(0.0);
					double max_p(0.0);
					int max_i(0);
					for(unsigned int m = 0; m < s0.size(); m++)
					{
						if(S[l] == s0[m])
						{
							tp[m] = truncate(tp[m],6);
							acc_p += tp[m];
							if(tp[m] > max_p)
							{
								max_p = tp[m];
								max_i = m;
							}
						}
					}
					double comp = 1 - acc_p;
					tp[max_i] += comp;

					//Truncate precision for the O-function
					acc_p = 0.0;
					max_p = 0.0;
					max_i = 0;
					for(unsigned int m = 0; m < ss.size(); m++)
					{
						if(S[l] == ss[m])
						{
							op[m] = truncate(op[m],6);
							acc_p += op[m];
							if(op[m] > max_p)
							{
								max_p = op[m];
								max_i = m;
							}
						}
					}
					comp = 1 - acc_p;
					op[max_i] += comp;
				}

				//Replace the truncated valuesin the T & O functions
				get<2>(T[k]) = tp;
				get<2>(O[k]) = op;
			}

			//Debug the generated POMDP
			if(!checkPomdp(S,A,Z,T,O))
			{
				cout << "Error while generating the base-POMDP for abstract state " + abs_s[j];
				cout << " at level " << i - 1 << "." << endl;
				return;
			}

			//Once the base POMDP for the j-th abstract state is defined, use it to
			//create an abstract action to transit from J to each of its neighbors
			for(unsigned int k = 0; k < neig_s.size(); k++)
			{
				//Modify the (*,terminate,*) transitions that start in goal peripheral states
				//so that they end in "absb-g"
				int term_index = distance(A.begin(),find(A.begin(),A.end(),"terminate"));
				vector<pMat> T_cp = T;
				for(unsigned int l = 0; l < get<0>(T_cp[term_index]).size(); l++)
				{
					ite = find(goal_s[k].begin(),goal_s[k].end(),get<0>(T_cp[term_index])[l]);
					if(ite != goal_s[k].end())
						get<1>(T_cp[term_index])[l] = "absb-g";
				}

				//Define the reward
				vector<string> G,P;
				vector< tuple<string,string> > Gpair,Ppair;

				//Define punish rewards so that the agent prefers to execute abstract
				//actions only from their intended S0 state
				if(i != bottom_level)
				{
					for(unsigned int l = 0; l < A.size(); l++)
					{
						//Decompose AA's to get their intended S0 by design
						vector<string> a_vec = splitStr(A[l],"->");
						if(a_vec.size() != 2) continue;
						for(unsigned int m = 0; m < S.size(); m++)
						{
							//-1 will be the R only for the intended S0 or 'absb'
							if(a_vec[0] == S[m] || "absb-g" == S[m] || "absb-ng" == S[m] || "extra" == S[m]) continue;

							//Punish executing the l-th abstract action from non-intended S0 abstract states and 'extra'
							tuple<string,string> pp_tmp(S[m],A[l]);
							Ppair.push_back(pp_tmp);
						}
					}
				}

				//Punish executing any action (that is diff. to "term") from the "extra" state
				for(unsigned int l = 0; l < A.size(); l++)
				{
					if(A[l] != "terminate")
						Ppair.push_back(tuple<string,string>("extra",A[l]));
					else continue;
				}

				//Executing "term" from non-peri-state is punished
				Ppair.push_back(tuple<string,string>("*","terminate"));

				//Rewards for the extra state
				Gpair.push_back(tuple<string,string>("extra","terminate"));// Rewarded to exec "term" from the extra state
				P.push_back("extra");// Transiting to the extra state is punished

				//Rewards for peri-states
				for(unsigned int l = 0; l < peri_s.size(); l++)
				{
					//Executing "term" from peri-states is rewarded
					Gpair.push_back(tuple<string,string>(peri_s[l],"terminate"));

					ite = find(goal_s[k].begin(),goal_s[k].end(),peri_s[l]);
					if(ite == goal_s[k].end())
					{
						//Transiting to non-goal peri-states is punished
						P.push_back(peri_s[l]);
					}

				}

				//Reward executing "term" from "absb-g"
				Gpair.push_back(tuple<string,string>("absb-g","terminate"));

				//Generate the (j->k) abstract action
				bool success;
				double discount = 0.95;
				vector<string> order;
				order.push_back("g");
				order.push_back("p");
				order.push_back("pp");
				order.push_back("gp");
				pomdp model = generatePomdp(S,A,Z,T_cp,O,G,Gpair,P,Ppair,discount,order,success);

				//Compute the abstract action's policy
				policy pol = solveModel(model,75,10,0.01);

				//Add the j-th abstract state and its neighbors in the fourth list
				vector<string> par_vec = neig_s;
				par_vec.push_back(abs_s[j]);
				get<4>(pol) = par_vec;

				//----------------------------------
				//STORE ABSTRACT ACTION

				//AA name
				string AA_name = abs_s[j] + "->" + neig_s[k];
				A_tmp.push_back(AA_name);

				//AA policy
				AA_.insert(pair<string,policy>(AA_name,pol));

				//Level in global vectors over which these Abs Actions are built upon
				unsigned gv_lvl = bottom_level - i;

				//AA T & O functions
				//tuple with a pair of pMats (T & O functions for abstract action)
				tuple<pMat,pMat> tmp_to = modelTO(model,abs_s[j],parent_peri_s,peri_s,pol,n_sim_aa_params,gv_lvl);

				//Generate the T & O probs. for the AA for other states as S0 (for T) & S1 (for O)
				for(unsigned int l = 0; l < abs_s.size(); l++)
				{
					if(abs_s[l] != abs_s[j])
					{
						//T prob. for l-th as starting state
						get<0>(get<0>(tmp_to)).push_back(abs_s[l]);
						get<1>(get<0>(tmp_to)).push_back(abs_s[l]);
						get<2>(get<0>(tmp_to)).push_back(1.0);
					}

					//O prob. for l-th as ending state
					ite = find(get<0>(get<1>(tmp_to)).begin(),get<0>(get<1>(tmp_to)).end(),abs_s[l]);
					if(ite == get<0>(get<1>(tmp_to)).end())
					{
						get<0>(get<1>(tmp_to)).push_back(abs_s[l]);
						get<1>(get<1>(tmp_to)).push_back(abs_s[l]);
						get<2>(get<1>(tmp_to)).push_back(1.0);
					}
				}

				//Save the AA T & O matrices
				T_tmp.push_back(get<0>(tmp_to));
				O_tmp.push_back(get<1>(tmp_to));

				//Add AA's states & observations
				if(k == 0)
				{
					//When the first A.A. is built, append the base-POMDP states and obs.
					//to the global vectors. This is done once due to all A.A. that start in
					//the j-th abs. state have the same S & Z
					vector<string> tmp_vec;
					tmp_vec.push_back(abs_s[j]);
					tmp_vec.insert(tmp_vec.end(),neig_s.begin(),neig_s.end());
					for(unsigned int l = 0; l < tmp_vec.size(); l++)
					{
						ite = find(S_tmp.begin(),S_tmp.end(),tmp_vec[l]);
						if(ite == S_tmp.end()) S_tmp.push_back(tmp_vec[l]);

						ite = find(Z_tmp.begin(),Z_tmp.end(),tmp_vec[l]);
						if(ite == Z_tmp.end()) Z_tmp.push_back(tmp_vec[l]);
					}

					//Check if the "none" observation has been added
					//ite = find(Z_tmp.begin(),Z_tmp.end(),string("none"));
					//if(ite == Z_tmp.end()) Z_tmp.push_back(string("none"));
				}
			}
		}

		//info
		cout << endl;

		//Save the abstract components for the level immediately above of the i-th level
		S_.push_back(S_tmp);
		A_.push_back(A_tmp);
		Z_.push_back(Z_tmp);
		T_.push_back(T_tmp);
		O_.push_back(O_tmp);

		//Create the AIToolbox model for the i-th level POMDP
		int idx = S_.size()-1;
		vector<string> gp,order;
		vector<tuple<string,string> > gpp;
		order.push_back("g");
		order.push_back("p");
		order.push_back("pp");
		order.push_back("gp");
		bool success;
		auto p = generatePomdp(S_[idx],A_[idx],Z_[idx],T_[idx],O_[idx],gp,gpp,gp,gpp,0.95,order,success);
		M_.push_back(get<0>(p));
	}

	//Build the global belief vector
	B_.clear();
	for(unsigned int i = 0; i < S_.size(); i++)
	{
		map<string,double> tmp;
		for(unsigned int j = 0; j < S_[i].size(); j++)
			tmp[S_[i][j]] = 0.0;

		B_.push_back(tmp);
	}

	//Display info on the constructed hierarchy of actions
	cout << "--------------------------------------" << endl;
	cout << ">> Done building hierarchy of actions." << endl;
	for(int i = S_.size() - 1; i >= 0 ; i--)
	{
		cout << "\tLEVEL-" << hs_.depth() - i - 1 << endl;
		cout << "\t\t# States: " << S_[i].size() << endl;
		cout << "\t\t# Observation: " << Z_[i].size() << endl;
		cout << "\t\t# Actions: " << A_[i].size() << endl;
	}
}

tuple<pMat,pMat> HPomdp::modelTO(pomdp const &p,string const &S0,vector<string> const &S1,vector<string> const &S1_peri,policy const &pol,int const &n_sim_aa_params,unsigned const &gv_lvl)
{
	//Invoke the policy simulator to obtain its transition probabilities
	pMat sim_tp = simPolicy(p,pol,S1_peri,n_sim_aa_params,gv_lvl);

	//Change the transition's state names
	unsigned int tp_size = get<0>(sim_tp).size();
	vector<string>::const_iterator ite;
	for(unsigned int i = 0; i < tp_size; i++)
	{
		//Change the starting state's name
		get<0>(sim_tp)[i] = S0;

		//Change the ending state's name
		ite = find(S1_peri.begin(),S1_peri.end(),get<1>(sim_tp)[i]);
		if(ite == S1_peri.end())
		{
			//The transition starts and ends in S0
			get<1>(sim_tp)[i] = S0;
		}
		else
		{
			//The transition ends in a peripheral state
			int index = distance(S1_peri.begin(),ite);
			get<1>(sim_tp)[i] = S1[index];
		}
	}

	//Delete those states that have 0 probability of being reached
	vector<int> z_st;
	for(unsigned int i = 0; i < tp_size; i++)//Get the zero-prob indexes
	{
		if(get<2>(sim_tp)[i] == 0) z_st.push_back(i);
	}
	for(int i = z_st.size() - 1; i >= 0; i--)//Delete the zero-prob transitions
	{
		get<0>(sim_tp).erase(get<0>(sim_tp).begin() + z_st[i]);
		get<1>(sim_tp).erase(get<1>(sim_tp).begin() + z_st[i]);
		get<2>(sim_tp).erase(get<2>(sim_tp).begin() + z_st[i]);
	}

	//Sum transitions that end in the same abstract state
	for(unsigned int i = 0; i < get<1>(sim_tp).size(); i++)
	{
		string s1 = get<1>(sim_tp)[i];
		vector<int> tmp;
		for(unsigned int j = i+1; j < get<1>(sim_tp).size(); j++)
			if(s1 == get<1>(sim_tp)[j]) tmp.push_back(j);

		for(int j = tmp.size()-1; j >= 0; j--)
		{
			//Add the prob. of the repeated ending state to its first appereance
			get<2>(sim_tp)[i] += get<2>(sim_tp)[tmp[j]];

			//Delete the repeated element
			get<0>(sim_tp).erase(get<0>(sim_tp).begin() + tmp[j]);
			get<1>(sim_tp).erase(get<1>(sim_tp).begin() + tmp[j]);
			get<2>(sim_tp).erase(get<2>(sim_tp).begin() + tmp[j]);
		}
	}

	//Create a pMat for the O-matrix
	//The obervation disttributions (at abstract levels) have no uncertainty, that is
	//a prob. of 1 is given
	pMat tmp_pm = sim_tp;
	for(unsigned int i = 0; i < get<0>(tmp_pm).size(); i++)
	{
		get<0>(tmp_pm)[i] = get<1>(tmp_pm)[i];
		get<2>(tmp_pm)[i] = 1.0;
	}

	//According the proposed methodology, at abstract levels T & O functions are the same
	tuple<pMat,pMat> sim_output(sim_tp,tmp_pm);

	return sim_output;
}

pMat HPomdp::simPolicy(pomdp const &p,policy const &pol,vector<string> const &peri, unsigned int const &sim_runs,unsigned const &gv_lvl)
{
	POMDP::Policy sim_pol = get<0>(pol);
	vector<string> S = get<1>(p);
	vector<string> A = get<2>(p);
	vector<string> Z = get<3>(p);
	int horizon = get<10>(p);

	//Get the full model for the 'gv_lvl' level
	POMDP::Model<MDP::Model> model = M_[gv_lvl];

	//Create a uniform distribution over the non-peripheral states
	int nps(0);
	POMDP::Belief global_b(S_[gv_lvl].size());
	vector<string>::const_iterator ite;
	for(unsigned int i = 0; i < S.size(); i++)
	{
		if(S[i] == "absb-g" || S[i] == "absb-ng" || S[i] == "extra") continue;

		ite = find(peri.begin(),peri.end(),S[i]);
		if(ite == peri.end()) nps++;
	}
	double nps_p =  1 / static_cast<double>(nps);
	for(unsigned int i = 0; i < S_[gv_lvl].size(); i++)
	{
		ite = find(S.begin(),S.end(),S_[gv_lvl][i]);

		//States in the abstract action
		if(ite != S.end())
		{
			ite = find(peri.begin(),peri.end(),S_[gv_lvl][i]);
			//non-peripheral states
			if(ite == peri.end()) global_b(i) = nps_p;
			//peripheral states
			else global_b(i) = 0.0;
		}
		//States outside the abstract action
		else global_b(i) = 0.0;
	}

	//Build actions mapping from local->global
	vector<string>::iterator ite2;
	vector<int> a_map(A.size(),0);
	for(unsigned int i = 0; i < A.size(); i++)
	{
		ite2 = find(A_[gv_lvl].begin(),A_[gv_lvl].end(),A[i]);
		if(ite2 != A_[gv_lvl].end())
			a_map[i] = distance(A_[gv_lvl].begin(),ite2);
		else a_map[i] = -1;
	}

	//Build belief vector mapping from global->local
	int extra_idx = distance(S.begin(),find(S.begin(),S.end(),"extra"));//Get the index of the 'extra' state
	vector<int> b_map(S_[gv_lvl].size(),0);
	for(unsigned int i = 0; i < S_[gv_lvl].size(); i++)
	{
		ite2 = find(S.begin(),S.end(),S_[gv_lvl][i]);
		//The i-th state is in the Abs. Action's state space
		if(ite2 != S.end())
			b_map[i] = distance(S.begin(),ite2);
		//The i-th state is NOT in the Abs. Action's state space
		else b_map[i] = extra_idx;
	}

	//Build observations mapping from global->local
	int z_extra_idx = distance(Z.begin(),find(Z.begin(),Z.end(),"extra"));//Get the index of the 'extra' observation
	vector<int> z_map(Z_[gv_lvl].size(),0);
	for(unsigned int i = 0; i < Z_[gv_lvl].size(); i++)
	{
		ite2 = find(Z.begin(),Z.end(),Z_[gv_lvl][i]);
		//The i-th observation is in the Abs. Action's observation space
		if(ite2 != Z.end())
			z_map[i] = distance(Z.begin(),ite2);
		//The i-th observation is NOT in the Abs. Action's observation space
		else z_map[i] = z_extra_idx;
	}

	//Vector for counting the final state
	vector<unsigned> fas(peri.size()+1,0);

	//Simulate the policy 'sim_runs' times
	std::default_random_engine rand(Impl::Seeder::getSeed());
	for(unsigned int i = 0; i < sim_runs; i++)
	{
		//Get a copy of the non-peripheral states initial belief
		POMDP::Belief b = global_b;

		//W: Obtaining the actual starting-state (The agent does not know this)
		size_t s = sampleProbability(S_[gv_lvl].size(),b,rand);

		//Build the initial local b-vec to select the first action
		POMDP::Belief loc_b(S.size());// inits values as 0.0
		for(unsigned int j = 0; j < S.size(); j++) loc_b(j) = 0.0;
		for(unsigned int j = 0; j < S_[gv_lvl].size(); j++)
			loc_b(b_map[j]) = b(j);

		//A: Get the first action
		auto a_id = sim_pol.sampleAction(loc_b, horizon);

		//Last visited state known by the AA
		size_t lks = b_map[s];

		//Start the execution for an amount of steps (horizon) the sim_pol was computed
		for(int t = horizon - 1; t >= 0; --t)
		{
			//A: get the current action
			size_t a = get<0>(a_id);

			//Sys: Evaluate if the current action is the termination-action
			if(A[a] == "terminate")
			{
				break;
			}

			//Map the action from local-> global
			a = a_map[a];

			//W: the world advances 1 step, given that "a" is performed from "s". This
			//   returns the resulting state, the generated observation and  the reward
			//   obtained.
			auto s1_o_r = model.sampleSOR(s, a);
			size_t s1 = get<0>(s1_o_r);
			size_t o = get<1>(s1_o_r);
			double r = get<2>(s1_o_r);

			//A: Update the belief
			b = AIToolbox::POMDP::updateBelief(model, b, a, o);

			//Map the belief vector from global to local
			loc_b = POMDP::Belief(S.size());
			if(z_map[o] == z_extra_idx)
			{
				for(unsigned int j = 0; j < S.size(); j++)
				{
					if(j == extra_idx) loc_b(j) = 1.0;
					else loc_b(j) = 0.0;
				}
			}
			else
			{
				for(unsigned int j = 0; j < S.size(); j++) loc_b(j) = 0.0;
				for(unsigned int j = 0; j < S_[gv_lvl].size(); j++)
					loc_b(b_map[j]) += b(j);
			}

			//Save the prev. visited state if it is in the AA's state space
			if(b_map[s1] != extra_idx) lks = b_map[s1];

			//W: Update the state of the world
			s = s1;

			//A: Find out what the next action should be
			if (t > (int)sim_pol.getH())
				a_id = sim_pol.sampleAction(loc_b, sim_pol.getH());
			//Do not go beyond the horizon for which the policy was computed
			else
			{
				o = z_map[o];//map the perceived observation
				a_id = sim_pol.sampleAction(std::get<1>(a_id), o, t);
			}
		}

		//Check for the agent's current state
		string final_s;
		if(b_map[s] != extra_idx) final_s = S[b_map[s]];
		else final_s = S[lks];

		//Check if the final state is a peripheral one, if not, then it
		//must  be in the starting abstract state
		ite = find(peri.begin(),peri.end(),final_s);
		if(ite == peri.end())
		{
			//Final state is in starting abstract state
			fas[fas.size()-1]++;
		}
		else
		{
			//Final state is a peripheral state
			int p_index = std::distance(peri.begin(),ite);
			fas[p_index]++;
		}
	}

	//Compute the transition probabilities based on the final-states count
	vector<string> abs_states = peri;
	abs_states.push_back("start_abs_state");
	vector<string> s0,s1;
	vector<double> tp;
	for(unsigned int i = 0; i < fas.size(); i++)
	{
		s0.push_back("start_abs_state");
		s1.push_back(abs_states[i]);

		//Compute the transition to the i-th abstract state
		double prob = static_cast<double>(fas[i]) / static_cast<double>(sim_runs);
		tp.push_back(prob);
	}

	//Make sure the distribution adds up to 1
	double total_p(0.0);
	double max_p(0.0);
	int max_id(-1);
	for(unsigned int i = 0; i < tp.size(); i++)
	{
		total_p += tp[i];
		if(tp[i] > max_p)
		{
			max_p = tp[i];
			max_id = i;
		}
	}
	tp[max_id] += (1 - total_p);

	return pMat(s0,s1,tp);
}

bool HPomdp::setSimulatorModel(string const &prob_file)
{
	//The  environment must be loaded in order to initialize the simulator
	if(!env_loaded_) return false;

	//Check the concrete model will be used (default option)
	if(prob_file == "use-concrete-model")
	{
		ws_T = T_[0];
		ws_O = O_[0];
		sim_initialized_ = true;

		return true;
	}
	else
	{
		//Load the JSON file
		ifstream infile(prob_file);
		if(!infile.is_open())
		{
			cout << ">> Error: could not open simulator probabilities file " + prob_file << endl;
			return false;
		}

		vector<string> sim_actions;
		vector<double> sim_T;
		vector<double> sim_O;
		vector< tuple<int,int,double> > sim_o_dist;

		//The JSON file is expected to have the same structure one 
		//generated by the one the "createProbFile" method
		try
		{
			//Parse the file into a JSON structure
			json jj;
			infile >> jj;
			infile.close();

			//List of concrete actions that operate at the bottom level in the hierarchy
			sim_actions.push_back("up");
			sim_actions.push_back("down");
			sim_actions.push_back("left");
			sim_actions.push_back("right");

			//Gather the transition and observation probabilities
			for(unsigned int i = 0; i < jj["Function"].size(); i++)
			{
				vector<double> tmp;
				vector< tuple<int,int,double> > sim_o_dist_tmp;

				//Gather the transition probabilities
				if(jj["Function"][i]["type"] == "T")
				{
					tmp.push_back(static_cast<double>(jj["Function"][i]["up"]));
					tmp.push_back(static_cast<double>(jj["Function"][i]["down"]));
					tmp.push_back(static_cast<double>(jj["Function"][i]["left"]));
					tmp.push_back(static_cast<double>(jj["Function"][i]["right"]));

					sim_T = tmp;
				}

				//Gather the observation probabilities
				if(jj["Function"][i]["type"] == "O")
				{
					tmp.push_back(static_cast<double>(jj["Function"][i]["precision"]));

					sim_O = tmp;
				}

				//Gather the observation probabilities as a distribution function
				if(jj["Function"][i]["type"] == "O-dist")
				{
					for(unsigned int j = 0; j < jj["Function"][i]["p-dist"].size(); j++)
					{
						int c_x = jj["Function"][i]["p-dist"][j]["x"];
						int c_y = jj["Function"][i]["p-dist"][j]["y"];
						double c_p = jj["Function"][i]["p-dist"][j]["prob"];
						tuple<int,int,double> c_tmp(c_x,c_y,c_p);
						sim_o_dist_tmp.push_back(c_tmp);
					}

					sim_o_dist = sim_o_dist_tmp;
				}
			}

			//Check for valid values
			if(sim_T.size() != 4 || sim_O.size() != 1)
			{
				return false;
			}
			for(unsigned int i = 0; i < sim_T.size(); i++)
			{
				bool fail(false);
				if(sim_T[i] < 0 || sim_T[i] > 1) fail = true;
				if(i == 0)
				{
					if(sim_O[i] < 0 || sim_O[i] > 1) fail = true;
				}

				if(fail)
				{
					return false;
				}
			}
			for(unsigned int i = 0; i < sim_o_dist.size(); i++)
			{
				double cell_prob = get<2>(sim_o_dist[i]);
				if(cell_prob < 0 || cell_prob > 1)
				{
					return false;
				}
			}
		}
		catch(std::exception &e)
		{
			cout << e.what() << endl;
			return false;
		}

		//Build the T & O functions for the world-simulator

		//Get all the states at the bottom of the hierarchy
		unsigned depth = hs_.depth();
		vector<string> con_S = hs_.keysAtLevel(depth - 1);

		//Determine if the the O-function will be built using the default prob. or the distribution
		bool use_dist = (sim_o_dist.size() > 0);

		//Set of concrete observations
		if(!use_dist) con_S.push_back("none");
		Z_.push_back(con_S);

		//Transition & Observation probability functions
		vector<pMat> bottom_T;
		vector<pMat> bottom_O;
		for(unsigned int i = 0; i < A_[0].size(); i++)
		{
			//T function params: starting-state. ending-state & transition-prob
			vector<string> s0,s1;
			vector<double> tp;

			//O function params: current-state, observation & observation-prob
			vector<string> cs, ob;
			vector<double> op;

			//Iterate over each state at the bottom of the hierarchy
			for(unsigned int j = 0; j < S_[0].size(); j++)
			{
				vector<string> neig;
				bool result;

				if(A_[0][i] == "up") neig = nh_.aboveOf(S_[0][j],result);
				else if(A_[0][i] == "down") neig = nh_.belowOf(S_[0][j],result);
				else if(A_[0][i] == "left") neig = nh_.leftOf(S_[0][j],result);
				else if(A_[0][i] == "right") neig = nh_.rightOf(S_[0][j],result);

				if(neig.size() > 0)
				{
					//Transit to the ending state

					//T
					s0.push_back(S_[0][j]);
					s1.push_back(neig[0]);
					tp.push_back(sim_T[i]);
					s0.push_back(S_[0][j]);
					s1.push_back(S_[0][j]);
					tp.push_back(1 - sim_T[i]);

					//O
					if(!use_dist)
					{
						cs.push_back(S_[0][j]);
						ob.push_back(neig[0]);
						op.push_back(sim_O[0]);
						cs.push_back(S_[0][j]);
						ob.push_back("none");
						op.push_back(1 - sim_O[0]);
					}
				}
				else
				{
					//Stay in the starting state

					//T
					s0.push_back(S_[0][j]);
					s1.push_back(S_[0][j]);
					tp.push_back(1);

					//O
					if(!use_dist)
					{
						cs.push_back(S_[0][j]);
						ob.push_back(S_[0][j]);
						op.push_back(sim_O[0]);//Precion of perceiving the observation
						cs.push_back(S_[0][j]);
						ob.push_back("none");
						op.push_back(1 - sim_O[0]);//Precion of perceiving the observation
					}
				}

				//O (probability distribution function)
				if(use_dist)
				{
					//Determine which cell will be the center of the kernel
					string center_cell;
					if(neig.size() > 0) center_cell = neig[0];
					else center_cell = S_[0][j];

					vector<string> cs_tmp, ob_tmp;
					vector<double> op_tmp;
					for(unsigned int k = 0; k < sim_o_dist.size(); k++)
					{
						int x = get<0>(sim_o_dist[k]);
						int y = get<1>(sim_o_dist[k]);
						double p = get<2>(sim_o_dist[k]);

						//Its the prob. of the central cell
						if(x == 0 && y == 0)
						{
							cs_tmp.push_back(S_[0][j]);
							ob_tmp.push_back(center_cell);
							op_tmp.push_back(p);
						}
						//A neighbor cell whose connectivity to the central cell 
						//must be checked before adding it as a possible observation
						else
						{
							int abs_x = abs(x);
							int abs_y = abs(y);
							string x_most,y_most,tgt_cell;

							vector<string> v_tmp;
							string curr_c(center_cell);
							bool cell_found(false);
							for(int l = 0; l < abs_x; l++)
							{
								if(x > 0) v_tmp = nh_.rightOf(curr_c,cell_found);
								else v_tmp = nh_.leftOf(curr_c,cell_found);

								if(!cell_found) break;
								else curr_c = v_tmp[0];
							}
							x_most = curr_c;

							//Check that the furthest X was reached
							if(!cell_found && abs_x > 0) continue;

							curr_c = center_cell;
							for(int l = 0; l < abs_y; l++)
							{
								if(y > 0) v_tmp = nh_.aboveOf(curr_c,cell_found);
								else v_tmp = nh_.belowOf(curr_c,cell_found);

								if(!cell_found) break;
								else curr_c = v_tmp[0];
							}
							y_most = curr_c;

							//Check that the furthest Y was reached
							if(!cell_found) continue;

							//Now check if the target-cell can be reached from both the  X & Y furthest
							//From furth-Y to target
							curr_c = y_most;
							for(int l = 0; l < abs_x; l++)
							{
								if(x > 0) v_tmp = nh_.rightOf(curr_c,cell_found);
								else v_tmp = nh_.leftOf(curr_c,cell_found);

								if(!cell_found) break;
								else curr_c = v_tmp[0];
							}

							//Check if target cell reached from the furthest Y
							if(!cell_found) continue;

							//From furth-X to target
							curr_c = x_most;
							for(int l = 0; l < abs_y; l++)
							{
								if(y > 0) v_tmp = nh_.aboveOf(curr_c,cell_found);
								else v_tmp = nh_.belowOf(curr_c,cell_found);

								if(!cell_found) break;
								else curr_c = v_tmp[0];
							}

							//The target cell can be reached from both paths,
							//& therefore it cen be perceived
							if(cell_found)
							{
								cs_tmp.push_back(S_[0][j]);
								ob_tmp.push_back(curr_c);
								op_tmp.push_back(p);

							}
						}
					}

					//Normalize the probabilitiy for the valid observations for the j-th state
					double total_p(0.0);
					double real_total_p(0.0);
					for(unsigned int k = 0; k < op_tmp.size(); k++) total_p += op_tmp[k];
					for(unsigned int k = 0; k < op_tmp.size(); k++)
					{
						op_tmp[k] /= total_p;
						real_total_p += op_tmp[k];
					}
					for(unsigned int k = 0; k < ob_tmp.size(); k++)
					{
						//Make sure the total probabilities sum 1
						if(ob_tmp[k] == cs_tmp[k])
						{
							//Add the remaining to the central observation
							op_tmp[k] += (1 - real_total_p);
							break;
						}
					}

					//Append the observation probabilities
					cs.insert(cs.end(),cs_tmp.begin(),cs_tmp.end());
					ob.insert(ob.end(),ob_tmp.begin(),ob_tmp.end());
					op.insert(op.end(),op_tmp.begin(),op_tmp.end());
				}
			}

			//Build i-th action's T & O function
			pMat a_T(s0,s1,tp);
			pMat a_O(cs,ob,op);

			//Save i-th action's functions
			bottom_T.push_back(a_T);
			bottom_O.push_back(a_O);
		}

		//Transition & Observation functions
		ws_T = bottom_T;
		ws_O = bottom_O;

		sim_initialized_ = true;

		return true;
	}
}

void HPomdp::updateBeliefDyn(string const &a, string const &z)
{
	//belief at t-1
	map<string,double> tmp = b_t_;

	//T-dist and O-dist for action 'a'
	pMat a_T, a_O;
	int idx = std::distance(A_[0].begin(),std::find(A_[0].begin(),A_[0].end(),a));
	a_T = T_[0][idx];
	a_O = O_[0][idx];

	//Set of states from which 'z'can be observed
	vector<string> z_states = z2s_[z];

	//Iterate over the possible ending states
	b_t_.clear();
	for(unsigned int i = 0; i < z_states.size(); i++)
	{
		//Get the prob. of perceiving O(z,i-th s,a)
		double o_prob = 0.0;
		bool zero_prob(true);
		for(unsigned int j = 0; j < std::get<0>(a_O).size(); j++)
		{
			if(z_states[i] == std::get<0>(a_O)[j])
			{
				if(z == std::get<1>(a_O)[j])
				{
					o_prob = std::get<2>(a_O)[j];
					zero_prob = false;
					break;
				}
			}
		}

		//If O(z,i-th s,a) == 0, there is no need in computing the sum
		//and, hence, should not be included in the next belief vector
		if(zero_prob) continue;

		//Iterate over those states in the b(t-1) belief vector
		double sum(0.0);
		for(map<string,double>::iterator ite = tmp.begin(); ite != tmp.end(); ++ite)
		{
			//Get the T(s0,s1,a) probability
			double prev_b_prob = ite->second;
			string s0 = ite->first;
			string s1 = z_states[i];
			double t_prob(0.0);
			for(unsigned int j = 0; j < std::get<0>(a_T).size(); j++)
			{
				if(std::get<0>(a_T)[j] == s0 && std::get<1>(a_T)[j] == s1)
				{
					t_prob = std::get<2>(a_T)[j];
					break;
				}
			}

			//T(s0,s1,a) * b_{t-1}(s0)
			double b_prob = t_prob * prev_b_prob;

			sum += b_prob;
		}

		//O(z,i-th s,a) * sum
		b_t_[z_states[i]] = o_prob * sum;
	}

	//Normalize the resulting belief distribution
	double total(0.0);
	for(map<string,double>::iterator ite = b_t_.begin(); ite != b_t_.end(); ++ite)
		total += ite->second;
	for(map<string,double>::iterator ite = b_t_.begin(); ite != b_t_.end(); ++ite)
		b_t_[ite->first] = ite->second / total;

	//Remove the remaining elements with 0 probability
	vector<string> vec;
	for(map<string,double>::iterator ite = b_t_.begin(); ite != b_t_.end(); ++ite)
		if(ite->second == 0.0) vec.push_back(ite->first);
	for(unsigned int i = 0; i < vec.size(); i++)
		b_t_.erase(vec[i]);

//Sum the probabilities upwards in the hierarchy of states
unsigned bottom_level = hs_.depth() - 1; //Get the ply at the hierarchical state space bottom level
for(unsigned int i = 0; i < B_.size(); i++)
{
	//Probabilities for concrete states
	if(i == 0)
	{
		//Set all concrete states to 0.0 prob
		for(map<string,double>::iterator ite = B_[0].begin(); ite != B_[0].end(); ++ite)
			ite->second = 0.0;

		//Set set concrete states with non-zero prob
		for(map<string,double>::iterator ite = b_t_.begin(); ite != b_t_.end(); ++ite)
			B_[0][ite->first] = ite->second;
	}
	//Probabilities for abstract states
	else
	{
		for(map<string,double>::iterator ite = B_[i].begin(); ite != B_[i].end(); ++ite)
		{
			//Sum the probs of all the children states of 'ite'
			vector<string> S = hs_.keysOfChildren(ite->first);
			double sum(0.0);
			for(unsigned int j = 0; j < S.size(); j++)
				sum += B_[i-1][S[j]];

			//Assign the prob
			ite->second = sum;
		}
	}
}
}

bool HPomdp::setSimulatorState(string const &world_state)
{
	if(!env_loaded_)
	{
		cout << ">> Error: the environment has not been loaded." <<  endl;
		return false;
	}
	vector<string>::iterator ite;
	ite = find(S_[0].begin(),S_[0].end(),world_state);
	if(ite == S_[0].end())
	{
		cout << ">> Error: unexisting state " + world_state + " for the simulator." <<  endl;
		return false;
	}
	else
	{
		//Set the concrete state
		ws_state = world_state;
		sim_state_set_ = true;

		//Reset the dynamic belief vector
		b_t_.clear();
		b_t_[world_state] = 1.0;
		string ch_s = world_state;
		for(unsigned int i = 0; i < B_.size(); i++)
		{
			for(map<string,double>::iterator ite = B_[i].begin(); ite != B_[i].end(); ++ite)
				ite->second = 0.0;

			if(i == 0) B_[i][world_state] = 1.0;
			else
			{
				bool success;
				string par_s = hs_.parent(ch_s,success);
				B_[i][par_s] = 1.0;

				ch_s = par_s;
			}
		}

		return true;
	}
}

string HPomdp::getSimulatorState()
{
	return ws_state;
}

void HPomdp::resetStepCounter()
{
	ca_count_ = 0;
	stuck = false;
}

int HPomdp::getStepCounter()
{
	return ca_count_;
}

string HPomdp::interactWithWorld(string const &a)
{
	//Make sure the state has been set before attempting to change it
	if(!sim_state_set_)
	{
		return string("~Error: the state of the world has not been set.");
	}

	//Make sure the requested action is actually modeled
	vector<string>::iterator ite;
	ite = find(A_[0].begin(),A_[0].end(),a);
	if(ite == A_[0].end())
	{
		return string("~Error: concrete action " + a + " does not exist.");
	}
	else
	{
		//Get the action index
		int ai = distance(A_[0].begin(),ite);

		//------------------------------------------------------------------
		// ENDING STATE SAMPLING

		//Build the T prob. distribution for the pair (ws_state - a)
		//Get the prob. distribution
		vector<string> s1;
		vector<double> tp;
		for(unsigned int i = 0; i < get<0>(ws_T[ai]).size(); i++)
		{
			if(get<0>(ws_T[ai])[i] == ws_state)
			{
				s1.push_back(get<1>(ws_T[ai])[i]);
				tp.push_back(get<2>(ws_T[ai])[i] * 1000000);
			}
		}

		//Build the cumulative dist. prob. of T for the pair (ws_state - a)
		for(unsigned int i = 0; i < tp.size(); i++)
		{
			if(i > 0) tp[i] += tp[i-1];
		}

		//Sample an ending state from the cumulative T function
		int sample_s1_index = rand() % 1000000;
		string sample_s1;
		for(unsigned int i = 0; i < tp.size(); i++)
		{
			if(sample_s1_index < tp[i])
			{
				sample_s1 = s1[i];
				break;
			}
		}

		//------------------------------------------------------------------
		// OBSERVATION SAMPLING
		
		//From the O-distribution of the smapled ending state, sample an observation
		vector<string> ob;
		vector<double> op;
		for(unsigned int i = 0; i < get<0>(ws_O[ai]).size(); i++)
		{
			if(get<0>(ws_O[ai])[i] == sample_s1)
			{
				ob.push_back(get<1>(ws_O[ai])[i]);
				op.push_back(get<2>(ws_O[ai])[i] * 1000000);
			}
		}
		for(unsigned int i = 0; i < op.size(); i++)
		{
			if(i > 0) op[i] += op[i-1];
		}
		int sample_ob_index = rand() % 1000000;
		string sample_ob;
		for(unsigned int i = 0; i < op.size(); i++)
		{
			if(sample_ob_index < op[i])
			{
				sample_ob = ob[i];
				break;
			}
		}

		//Update the world's state
		ws_state = sample_s1;

		//Update the concrete action counter
		ca_count_++;

		//Return the perceived observation
		return sample_ob;
	}
}

pomdp HPomdp::concModel()
{
	//Get the concrete full model
	vector<string> S,A,Z;
	vector<pMat> T,O;
	S = S_[0];
	A = A_[0];
	Z = Z_[0];
	T = T_[0];
	O = O_[0];

	//Add the components related to the "terminate"
	vector<string> s_tmp;
	vector<double> p_tmp;
	S.push_back("absb");
	A.push_back("terminate");
	Z.push_back("none");
	T.push_back(pMat(s_tmp,s_tmp,p_tmp));
	O.push_back(pMat(s_tmp,s_tmp,p_tmp));
	//T & O for ternitae action
	for(unsigned int i = 0; i < S.size(); i++)
	{
		get<0>(T[T.size()-1]).push_back(S[i]);
		get<1>(T[T.size()-1]).push_back("absb");
		get<2>(T[T.size()-1]).push_back(1.0);

		get<0>(O[O.size()-1]).push_back(S[i]);
		get<1>(O[O.size()-1]).push_back("none");
		get<2>(O[O.size()-1]).push_back(1.0);
	}

	//Add the T & O for executing non-termination actions from "absb"
	for(unsigned int i = 0; i < A.size(); i++)
	{
		if(A[i] != "terminate")
		{
			get<0>(T[i]).push_back("absb");
			get<1>(T[i]).push_back("absb");
			get<2>(T[i]).push_back(1.0);

			get<0>(O[i]).push_back("absb");
			get<1>(O[i]).push_back("none");
			get<2>(O[i]).push_back(1.0);
		}
	}

	//Truncate probabilities' precision in order to avoid precision errors
	//and add the comp. prob to the most likely transition
	for(unsigned int j = 0; j < A.size(); j++)
	{
		//"terminate" action needs no truncation
		if(A[j] == "terminate") continue;

		//truncate the probabilities
		for(unsigned int k = 0; k < get<2>(T[j]).size(); k++) get<2>(T[j])[k] = truncate(get<2>(T[j])[k],6);
		for(unsigned int k = 0; k < get<2>(O[j]).size(); k++) get<2>(O[j])[k] = truncate(get<2>(O[j])[k],6);

		//Add the complement prob. for T
		vector<string> s0 = get<0>(T[j]);
		vector<double> tp = get<2>(T[j]);
		for(unsigned int k = 0; k < S.size(); k++)
		{
			double tot_p(0.0);
			double max_p(0.0);
			int max_i(-1);
			for(unsigned int l = 0; l < s0.size(); l++)
			{
				if(S[k] == s0[l])
				{
					tot_p += tp[l];
					if(tp[l] > max_p)
					{
						max_p = tp[l];
						max_i = l;
					}
				}
			}
			double comp_p = 1 - tot_p;
			tp[max_i] += comp_p;
		}
		get<2>(T[j]) = tp;

		//Add the complement prob. for O
		vector<string> ss = get<0>(O[j]);
		vector<double> op = get<2>(O[j]);
		for(unsigned int k = 0; k < S.size(); k++)
		{
			double tot_p(0.0);
			double max_p(0.0);
			int max_i(-1);
			for(unsigned int l = 0; l < ss.size(); l++)
			{
				if(S[k] == ss[l])
				{
					tot_p += op[l];
					if(op[l] > max_p)
					{
						max_p = op[l];
						max_i = l;
					}
				}
			}
			double comp_p = 1 - tot_p;
			op[max_i] += comp_p;
		}
		get<2>(O[j]) = op;
	}

	//Check for a valid POMDP
	if(!checkPomdp(S,A,Z,T,O))
	{
		cout << ">> Error[concModel]: Invalid POMDP for full-concrete model." << endl;

		POMDP::Model<MDP::Model> m(1,1,1);
		vector<tuple<string,string> > v_tmp;
		vector<pMat> pm_tmp;
		unsigned hh(0);
		pomdp full_p(m,s_tmp,s_tmp,s_tmp,pm_tmp,pm_tmp,s_tmp,v_tmp,s_tmp,v_tmp,hh);
		return full_p;
	}

	//Return the full POMDP
	POMDP::Model<MDP::Model> m(1,1,1);
	vector<tuple<string,string> > v_tmp;
	unsigned hh(0);
	pomdp full_p(m,S,A,Z,T,O,s_tmp,v_tmp,s_tmp,v_tmp,hh);
	return full_p;
}

policy HPomdp::solveModel(pomdp &p,size_t nBeliefs,unsigned horizon,double epsilon)
{
	//Solving POMDP  parameters
	auto m = get<0>(p);
	m.setDiscount(0.95);

	//Save the horizon used to compute its policy
	get<10>(p) = horizon;

	//Solve
	AIToolbox::POMDP::PBVI solver(nBeliefs, horizon, epsilon);
	auto solution = solver(m);

	//Create policy from the value function
	size_t num_S = get<1>(p).size();
	size_t num_A = get<2>(p).size();
	size_t num_Z = get<3>(p).size();
	POMDP::Policy pol(num_S,num_A,num_Z,get<1>(solution));

	return policy(pol,get<2>(p),get<3>(p),get<1>(p),get<1>(p),m);
}

map<string,policy> HPomdp::srPolicies()
{
	//Get list of buildings
	vector<string> all_b = hs_.keysAtLevel(1);

	//Get the list of all cells
	vector<string> cells = hs_.keysAtLevel(4);

	//List of policies to transit between sub-regions
	map<string,policy> pols;

	//Compute a policy from the i-th building to transit to each of its neighbors
	bool tmp_res(false);
	for(unsigned int i = 0; i < all_b.size(); i++)
	{
		//Get the neighbors of the i-th building
		vector<string> neig = nh_.neigTo(all_b[i],tmp_res);

		//Get the concrete states in the i-th building
		vector<string> i_cell;
		for(unsigned int j = 0; j < cells.size(); j++)
		{
			if(hs_.isAncestor(all_b[i],cells[j])) i_cell.push_back(cells[j]);
		}

		//Get the peri-states
		vector<string> peri;
		for(unsigned int j = 0; j < i_cell.size(); j++)
		{
			vector<string> tmp_p = nh_.neigToExc(i_cell[j],i_cell);
			peri.insert(peri.end(),tmp_p.begin(),tmp_p.end());
		}

		//Separate the peri states by the building to which they belong
		vector<vector<string> > n_peri(neig.size(),vector<string>());
		for(unsigned int j = 0; j < neig.size(); j++)
		{
			for(unsigned int k = 0; k < peri.size(); k++)
			{
				if(hs_.isAncestor(neig[j],peri[k])) n_peri[j].push_back(peri[k]);
			}
		}

		//Build the base POMDP for the i-th building
		vector<string> S,A,Z;
		vector<pMat> T,O;

		//S
		S = peri;
		S.insert(S.end(),i_cell.begin(),i_cell.end());

		//A
		A = A_[0];

		//Z
		vector<string>::iterator ite;
		for(unsigned int j = 0; j < S.size(); j++)
		{
			for(unsigned int k = 0; k < O_[0].size(); k++)
			{
				vector<string> ss = get<0>(O_[0][k]);
				vector<string> ob = get<1>(O_[0][k]);
				for(unsigned int l = 0; l < ss.size(); l++)
				{
					if(S[j] == ss[l])
					{
						ite = find(Z.begin(),Z.end(),ob[l]);
						if(ite == Z.end()) Z.push_back(ob[l]);
					}
				}
			}
		}

		//T & O
		for(unsigned int j = 0; j < A.size(); j++)
		{
			vector<string> s_tmp;
			vector<double> d_tmp;
			pMat t_tmp(s_tmp,s_tmp,d_tmp);
			pMat o_tmp(s_tmp,s_tmp,d_tmp);

			//From all the probabilities select those  for the elements in S, A & Z
			vector<string> s0 = get<0>(T_[0][j]);
			vector<string> s1 = get<1>(T_[0][j]);
			vector<double> tp = get<2>(T_[0][j]);
			vector<string> ss = get<0>(O_[0][j]);
			vector<string> ob = get<1>(O_[0][j]);
			vector<double> op = get<2>(O_[0][j]);

			//T
			vector<vector<int> > pti(peri.size(),vector<int>());
			for(unsigned int k = 0; k < s0.size(); k++)
			{
				ite = find(S.begin(),S.end(),s0[k]);
				if(ite == S.end()) continue;
				ite = find(S.begin(),S.end(),s1[k]);
				if(ite == S.end()) continue;
				get<0>(t_tmp).push_back(s0[k]);
				get<1>(t_tmp).push_back(s1[k]);
				get<2>(t_tmp).push_back(tp[k]);

				//check if the k-th transition starts in a peri-state, this will be used later
				//to complete the peri-state's T-distribution
				ite = find(peri.begin(),peri.end(),s0[k]);
				if(ite != peri.end())
				{
					int dd = distance(peri.begin(),ite);
					pti[dd].push_back(get<2>(t_tmp).size()-1);
				}
			}

			//Complete the T-distribution for the peri states
			for(unsigned int k = 0; k < pti.size(); k++)
			{
				double tot_p(0.0);
				double max_p(0.0);
				int max_i(-1);
				for(unsigned int l = 0; l < pti[k].size(); l++)
				{
					tot_p += get<2>(t_tmp)[pti[k][l]];
					if(get<2>(t_tmp)[pti[k][l]] > max_p)
					{
						max_p = get<2>(t_tmp)[pti[k][l]];
						max_i = l;
					}
				}

				//compute the complement prob
				double comp_p = 1 - tot_p;

				//Add the complement to the most likely S1
				get<2>(t_tmp)[pti[k][max_i]] += comp_p;
			}

			//O
			for(unsigned int k = 0; k < ss.size(); k++)
			{
				ite = find(S.begin(),S.end(),ss[k]);
				if(ite == S.end()) continue;
				get<0>(o_tmp).push_back(ss[k]);
				get<1>(o_tmp).push_back(ob[k]);
				get<2>(o_tmp).push_back(op[k]);
			}

			//Store the T & O matrices for the j-th action
			T.push_back(t_tmp);
			O.push_back(o_tmp);
		}

		//Add the components related to the "terminate"
		vector<string> s_tmp;
		vector<double> p_tmp;
		S.push_back("absb");
		A.push_back("terminate");
		Z.push_back("none");
		T.push_back(pMat(s_tmp,s_tmp,p_tmp));
		O.push_back(pMat(s_tmp,s_tmp,p_tmp));

		//Add the T & O for executing termination action
		for(unsigned int j = 0; j < S.size(); j++)
		{
			get<0>(T[T.size()-1]).push_back(S[j]);
			get<1>(T[T.size()-1]).push_back("absb");
			get<2>(T[T.size()-1]).push_back(1.0);

			get<0>(O[O.size()-1]).push_back(S[j]);
			get<1>(O[O.size()-1]).push_back("none");
			get<2>(O[O.size()-1]).push_back(1.0);
		}

		//Add the T & O for executing non-termination actions from "absb"
		for(unsigned int j = 0; j < A.size(); j++)
		{
			if(A[j] != "terminate")
			{
				get<0>(T[j]).push_back("absb");
				get<1>(T[j]).push_back("absb");
				get<2>(T[j]).push_back(1.0);

				get<0>(O[j]).push_back("absb");
				get<1>(O[j]).push_back("none");
				get<2>(O[j]).push_back(1.0);
			}
		}

		//Truncate probabilities' precision in order to avoid precision errors
		//and add the comp. prob to the most likely transition
		for(unsigned int j = 0; j < A.size(); j++)
		{
			//"terminate" action needs no truncation
			if(A[j] == "terminate") continue;

			//truncate the probabilities
			for(unsigned int k = 0; k < get<2>(T[j]).size(); k++) get<2>(T[j])[k] = truncate(get<2>(T[j])[k],6);
			for(unsigned int k = 0; k < get<2>(O[j]).size(); k++) get<2>(O[j])[k] = truncate(get<2>(O[j])[k],6);

			//Add the complement prob. for T
			vector<string> s0 = get<0>(T[j]);
			vector<double> tp = get<2>(T[j]);
			for(unsigned int k = 0; k < S.size(); k++)
			{
				double tot_p(0.0);
				double max_p(0.0);
				int max_i(-1);
				for(unsigned int l = 0; l < s0.size(); l++)
				{
					if(S[k] == s0[l])
					{
						tot_p += tp[l];
						if(tp[l] > max_p)
						{
							max_p = tp[l];
							max_i = l;
						}
					}
				}
				double comp_p = 1 - tot_p;
				tp[max_i] += comp_p;
			}
			get<2>(T[j]) = tp;

			//Add the complement prob. for O
			vector<string> ss = get<0>(O[j]);
			vector<double> op = get<2>(O[j]);
			for(unsigned int k = 0; k < S.size(); k++)
			{
				double tot_p(0.0);
				double max_p(0.0);
				int max_i(-1);
				for(unsigned int l = 0; l < ss.size(); l++)
				{
					if(S[k] == ss[l])
					{
						tot_p += op[l];
						if(op[l] > max_p)
						{
							max_p = op[l];
							max_i = l;
						}
					}
				}
				double comp_p = 1 - tot_p;
				op[max_i] += comp_p;
			}
			get<2>(O[j]) = op;
		}

		//Check for a valid POMDP
		if(!checkPomdp(S,A,Z,T,O))
		{
			cout << ">> Error[srPolicies]: Invalid POMDP for building " + all_b[i] << endl;
			return pols;
		}

		//Once the base POMDP for the i-th building (sub-region) is ready
		//compute the policy to travel to each of its beighbors
		for(unsigned int j = 0; j < neig.size(); j++)
		{
			//Define the reward
			vector<string> G,P;
			vector< tuple<string,string> > Gpair,Ppair;

			//Executing "term" from non-peri-state is punished
			Ppair.push_back(tuple<string,string>("*","terminate"));

			//Rewards for goal peri-states
			for(unsigned int k = 0; k < n_peri.size(); k++)
			{
				//Executing "term" from goal peri-states is rewarded
				if(j == k)
				{
					for(unsigned int l = 0; l < n_peri[k].size(); l++)
					{
						Gpair.push_back(tuple<string,string>(n_peri[k][l],"terminate"));
					}
				}
				//Transiting to non-goal peri-states is punished
				else
				{
					for(unsigned int l = 0; l < n_peri[k].size(); l++)
					{
						P.push_back(n_peri[k][l]);
					}
				}
			}

			//Generate the (j->k) abstract action
			bool success;
			double discount = 0.95;
			vector<string> order;
			order.push_back("g");
			order.push_back("p");
			order.push_back("pp");
			order.push_back("gp");
			pomdp model = generatePomdp(S,A,Z,T,O,G,Gpair,P,Ppair,discount,order,success);

			//Compute the abstract action's policy
			policy pol = solveModel(model,200,10,0.1);

			//Create the policy's name
			string pol_name = all_b[i] + neig[j];

			//Store the computed policy
			pols.insert(pair<string,policy>(pol_name,pol));
		}

	}

	return pols;
}

policy HPomdp::finalPol(string const &final_sr,string const &goal_s)
{
	//Get list of buildings
	vector<string> all_b = hs_.keysAtLevel(1);

	//Get the list of all cells
	vector<string> cells = hs_.keysAtLevel(4);

	//Compute a policy from the i-th building to transit to  each of its neighbors
	bool tmp_res(false);
	for(unsigned int i = 0; i < all_b.size(); i++)
	{
		if(all_b[i] != final_sr) continue;

		//Get the neighbors of the i-th building
		vector<string> neig = nh_.neigTo(all_b[i],tmp_res);

		//Get the concrete states in the i-th building
		vector<string> i_cell;
		for(unsigned int j = 0; j < cells.size(); j++)
		{
			if(hs_.isAncestor(all_b[i],cells[j])) i_cell.push_back(cells[j]);
		}

		//Get the peri-states
		vector<string> peri;
		for(unsigned int j = 0; j < i_cell.size(); j++)
		{
			vector<string> tmp_p = nh_.neigToExc(i_cell[j],i_cell);
			peri.insert(peri.end(),tmp_p.begin(),tmp_p.end());
		}

		//Separate the peri states by the building to which they belong
		vector<vector<string> > n_peri(neig.size(),vector<string>());
		for(unsigned int j = 0; j < neig.size(); j++)
		{
			for(unsigned int k = 0; k < peri.size(); k++)
			{
				if(hs_.isAncestor(neig[j],peri[k])) n_peri[j].push_back(peri[k]);
			}
		}

		//Build the base POMDP for the i-th building
		vector<string> S,A,Z;
		vector<pMat> T,O;

		//S
		S = peri;
		S.insert(S.end(),i_cell.begin(),i_cell.end());

		//A
		A = A_[0];

		//Z
		vector<string>::iterator ite;
		for(unsigned int j = 0; j < S.size(); j++)
		{
			for(unsigned int k = 0; k < O_[0].size(); k++)
			{
				vector<string> ss = get<0>(O_[0][k]);
				vector<string> ob = get<1>(O_[0][k]);
				for(unsigned int l = 0; l < ss.size(); l++)
				{
					if(S[j] == ss[l])
					{
						ite = find(Z.begin(),Z.end(),ob[l]);
						if(ite == Z.end()) Z.push_back(ob[l]);
					}
				}
			}
		}

		//T & O
		for(unsigned int j = 0; j < A.size(); j++)
		{
			vector<string> s_tmp;
			vector<double> d_tmp;
			pMat t_tmp(s_tmp,s_tmp,d_tmp);
			pMat o_tmp(s_tmp,s_tmp,d_tmp);

			//From all the probabilities select those  for the elements in S, A & Z
			vector<string> s0 = get<0>(T_[0][j]);
			vector<string> s1 = get<1>(T_[0][j]);
			vector<double> tp = get<2>(T_[0][j]);
			vector<string> ss = get<0>(O_[0][j]);
			vector<string> ob = get<1>(O_[0][j]);
			vector<double> op = get<2>(O_[0][j]);

			//T
			vector<vector<int> > pti(peri.size(),vector<int>());
			for(unsigned int k = 0; k < s0.size(); k++)
			{
				ite = find(S.begin(),S.end(),s0[k]);
				if(ite == S.end()) continue;
				ite = find(S.begin(),S.end(),s1[k]);
				if(ite == S.end()) continue;
				get<0>(t_tmp).push_back(s0[k]);
				get<1>(t_tmp).push_back(s1[k]);
				get<2>(t_tmp).push_back(tp[k]);

				//check if the k-th transition starts in a peri-state, this will be used later
				//to complete the peri-state's T-distribution
				ite = find(peri.begin(),peri.end(),s0[k]);
				if(ite != peri.end())
				{
					int dd = distance(peri.begin(),ite);
					pti[dd].push_back(get<2>(t_tmp).size()-1);
				}
			}

			//Complete the T-distribution for the peri states
			for(unsigned int k = 0; k < pti.size(); k++)
			{
				double tot_p(0.0);
				double max_p(0.0);
				int max_i(-1);
				for(unsigned int l = 0; l < pti[k].size(); l++)
				{
					tot_p += get<2>(t_tmp)[pti[k][l]];
					if(get<2>(t_tmp)[pti[k][l]] > max_p)
					{
						max_p = get<2>(t_tmp)[pti[k][l]];
						max_i = l;
					}
				}

				//compute the complement prob
				double comp_p = 1 - tot_p;

				//Add the complement to the most likely S1
				get<2>(t_tmp)[pti[k][max_i]] += comp_p;
			}

			//O
			for(unsigned int k = 0; k < ss.size(); k++)
			{
				ite = find(S.begin(),S.end(),ss[k]);
				if(ite == S.end()) continue;
				get<0>(o_tmp).push_back(ss[k]);
				get<1>(o_tmp).push_back(ob[k]);
				get<2>(o_tmp).push_back(op[k]);
			}

			//Store the T & O matrices for the j-th action
			T.push_back(t_tmp);
			O.push_back(o_tmp);
		}

		//Add the components related to the "terminate"
		vector<string> s_tmp;
		vector<double> p_tmp;
		S.push_back("absb");
		A.push_back("terminate");
		Z.push_back("none");
		T.push_back(pMat(s_tmp,s_tmp,p_tmp));
		O.push_back(pMat(s_tmp,s_tmp,p_tmp));

		//Add the T & O for executing termination action
		for(unsigned int j = 0; j < S.size(); j++)
		{
			get<0>(T[T.size()-1]).push_back(S[j]);
			get<1>(T[T.size()-1]).push_back("absb");
			get<2>(T[T.size()-1]).push_back(1.0);

			get<0>(O[O.size()-1]).push_back(S[j]);
			get<1>(O[O.size()-1]).push_back("none");
			get<2>(O[O.size()-1]).push_back(1.0);
		}

		//Add the T & O for executing non-termination actions from "absb"
		for(unsigned int j = 0; j < A.size(); j++)
		{
			if(A[j] != "terminate")
			{
				get<0>(T[j]).push_back("absb");
				get<1>(T[j]).push_back("absb");
				get<2>(T[j]).push_back(1.0);

				get<0>(O[j]).push_back("absb");
				get<1>(O[j]).push_back("none");
				get<2>(O[j]).push_back(1.0);
			}
		}

		//Truncate probabilities' precision in order to avoid precision errors
		//and add the comp. prob to the most likely transition
		for(unsigned int j = 0; j < A.size(); j++)
		{
			//"terminate" action needs no truncation
			if(A[j] == "terminate") continue;

			//truncate the probabilities
			for(unsigned int k = 0; k < get<2>(T[j]).size(); k++) get<2>(T[j])[k] = truncate(get<2>(T[j])[k],6);
			for(unsigned int k = 0; k < get<2>(O[j]).size(); k++) get<2>(O[j])[k] = truncate(get<2>(O[j])[k],6);

			//Add the complement prob. for T
			vector<string> s0 = get<0>(T[j]);
			vector<double> tp = get<2>(T[j]);
			for(unsigned int k = 0; k < S.size(); k++)
			{
				double tot_p(0.0);
				double max_p(0.0);
				int max_i(-1);
				for(unsigned int l = 0; l < s0.size(); l++)
				{
					if(S[k] == s0[l])
					{
						tot_p += tp[l];
						if(tp[l] > max_p)
						{
							max_p = tp[l];
							max_i = l;
						}
					}
				}
				double comp_p = 1 - tot_p;
				tp[max_i] += comp_p;
			}
			get<2>(T[j]) = tp;

			//Add the complement prob. for O
			vector<string> ss = get<0>(O[j]);
			vector<double> op = get<2>(O[j]);
			for(unsigned int k = 0; k < S.size(); k++)
			{
				double tot_p(0.0);
				double max_p(0.0);
				int max_i(-1);
				for(unsigned int l = 0; l < ss.size(); l++)
				{
					if(S[k] == ss[l])
					{
						tot_p += op[l];
						if(op[l] > max_p)
						{
							max_p = op[l];
							max_i = l;
						}
					}
				}
				double comp_p = 1 - tot_p;
				op[max_i] += comp_p;
			}
			get<2>(O[j]) = op;
		}

		//Check for a valid POMDP
		if(!checkPomdp(S,A,Z,T,O))
		{
			cout << ">> Error[finalPol]: Invalid POMDP for building " + all_b[i] << endl;
			POMDP::Policy pp(1,1,1);
			vector<string> ss;
			POMDP::Model<MDP::Model> mm(1,1,1);
			policy tmp_pol(pp,ss,ss,ss,ss,mm);
			return tmp_pol;
		}

		//Define the reward
		vector<string> G,P;
		vector< tuple<string,string> > Gpair,Ppair;

		//Executing "term" from non-peri-state is punished
		Ppair.push_back(tuple<string,string>("*","terminate"));

		//Transiting to peri-states is punished
		for(unsigned int j = 0; j < peri.size(); j++) P.push_back(peri[j]);

		//Execution "termminate" from the goal state is rewarded
		Gpair.push_back(tuple<string,string>(goal_s,"terminate"));

		//Generate the model
		bool success;
		double discount = 0.95;
		vector<string> order;
		order.push_back("g");
		order.push_back("p");
		order.push_back("pp");
		order.push_back("gp");
		pomdp model = generatePomdp(S,A,Z,T,O,G,Gpair,P,Ppair,discount,order,success);

		//Compute the abstract action's policy
		policy pol = solveModel(model,200,10,0.1);
		return pol;
	}
}

tuple<string,int> HPomdp::exePol(policy P,int max_s,string s0)
{
	vector<string> A = get<1>(P);
	vector<string> Z = get<2>(P);
	vector<string> S = get<3>(P);

	//Create initial belief state
	POMDP::Belief b(S.size());
	if(s0 == "not-state")
	{
		//Create a uniform distribution over the states, excluding "absb"
		double uni_p = 1 / static_cast<double>(S.size() - 1);
		for(unsigned int i = 0; i < S.size(); i++)
		{
			if(S[i] != "absb") b(i) = uni_p;
			else b(i) = 0;
		}
	}
	else
	{
		for(unsigned int i = 0; i < S.size(); i++)
		{
			if(S[i] == s0) b(i) = 1;
			else b(i) = 0;
		}
	}

	//Start executing the policy
	vector<string>::iterator ite_s;
	POMDP::Model<MDP::Model> pomdp_model = get<5>(P);
	POMDP::Policy pomdp_pol = get<0>(P);
	int horizon = pomdp_pol.getH();
	int step_count(0);
	string term("");
	while(true)
	{
		auto a_id = pomdp_pol.sampleAction(b, horizon);
		for(int t = horizon - 1; t >= 0; --t)
		{
			size_t ai = get<0>(a_id);

			if(A[ai] == "terminate")
			{
				term = "terminate";
				break;
			}

			//Save state, in case  of invalid observation
			//string curr_s = getSimulatorState();

			//Execute the selected action
			string obs;
			obs = interactWithWorld(A[ai]);

			//Get the received observation index
			ite_s = find(Z.begin(),Z.end(),obs);
			if(ite_s == Z.end())
			{
				//For invalid observation (the agent is out of range
				//of the current policy) end execution
				return tuple<string,int>(string("max_s"),0);
			}
			size_t oi = static_cast<size_t>(distance(Z.begin(),ite_s));

			//Update the belief vector
			b = AIToolbox::POMDP::updateBelief(pomdp_model, b, ai, oi);

			//Find out what the next action should be
			if (t > (int)pomdp_pol.getH())
				a_id = pomdp_pol.sampleAction(b, pomdp_pol.getH());
			else
				a_id = pomdp_pol.sampleAction(std::get<1>(a_id), oi, t);

			//Increase the step counter
			step_count++;
		}

		//Evaluate for maximum steps allowed
		if(step_count >= max_s)
		{
			term = "max_s";
			break;
		}
		//Evaluate if it finished by term action
		else if(term == "terminate") break;
	}

	//Save the most likely state at  the termination of the policy in the global variable "mls"
	double max_p(0.0);
	int max_i(0);
	for(unsigned int i = 0; i < S.size(); i++)
	{
		if(b(i) > max_p)
		{
			max_p = b(i);
			max_i = i;
		}
	}
	mls = S[max_i];

	tuple<string,int> result(term,step_count);
	return result;
}

void HPomdp::uncertainS()
{
	//Set the dynamic belief vector to a uniform dist. (full uncertainty)
	b_t_.clear();
	double n_conc_s(S_[0].size());
	double uni_p = static_cast<double>(1.0) / n_conc_s;
	for(unsigned int i = 0; i < S_[0].size(); i++) b_t_[S_[0][i]] = uni_p;

	//Set the hierarchical belief vector to a uniform distribution at each level
	for(unsigned int i = 0; i < S_.size(); i++)
	{
		n_conc_s = static_cast<double>(S_[i].size());
		uni_p = static_cast<double>(1.0) / n_conc_s;
		for(unsigned int j = 0; j < S_[i].size(); j++)
			B_[i][S_[i][j]] = uni_p;
	}
}

int HPomdp::getSizeS(int const &i)
{
	if(i < 0 || i >= S_.size()) return -1;
	else return S_[i].size();
}

bool HPomdp::hPolPlan(string const &gs,bool const &debug)
{
	//Hierarchical policy as a vector of local policies
	vector<policy> h_pol;

	//Verify that the goal state exists in the hierarchy
	if(hs_.ply(gs) < 0) return false;

	//Get the hierarchical state of the goal state
	vector<string> h_gs = hs_.hieState(gs);

	//Make the hierarchical policy start at the highest level, in the hierarchy of
	//states, that has more than 1 abstract state
	int top_diff_lvl(-1);
	for(unsigned int i = 0; i < hs_.depth(); i++)
	{
		auto tmp = hs_.keysAtLevel(i);
		if(tmp.size() > 1)
		{
			top_diff_lvl = i;
			break;
		}
	}

	//----------------------------------------------------------------------------
	// BUILD H-POLICY

	for(unsigned int i = top_diff_lvl - 1; i < (h_gs.size() - 1); i++)
	{
		//Parent state that bounds the state-space of each local policy
		string parent_node(h_gs[i]);

		//Index in global vectors for level "i"
		int gvi = hs_.depth() - i - 1;

		if(debug)
		{
			cout << "-------------------------" << endl;
			cout << "Parent node: " + parent_node << endl;
		}

		//S,A,Z,T,O
		vector<string> S,A,Z;
		vector<pMat> T,O;
		vector<string>::iterator ite;

		//-----------------------------
		//S: children & not-children neighbor states
		if((i+1) != top_diff_lvl)
		{
			S = hs_.keysOfChildren(parent_node);
			vector<string> peri;
			for(unsigned int j = 0; j < S.size(); j++)
			{
				vector<string> tmp = nh_.neigToExc(S[j],S);
				for(unsigned int k = 0; k < tmp.size(); k++)
				{
					ite = find(peri.begin(),peri.end(),tmp[k]);
					if(ite == peri.end()) peri.push_back(tmp[k]);
				}
			}
			S.insert(S.end(),peri.begin(),peri.end());

			//Add the 'extra'
			S.push_back("extra");
		}
		//Loc-pol at the highest level with more than 1 state
		else S = hs_.keysAtLevel(top_diff_lvl);

		//-----------------------------
		//A: actions that start and end in a pair of states in S
		for(unsigned int j = 0; j < A_[gvi-1].size(); j++)
		{
			vector<string> s0 = get<0>(T_[gvi-1][j]);
			vector<string> s1 = get<1>(T_[gvi-1][j]);
			vector<double> tp = get<2>(T_[gvi-1][j]);
			for(unsigned int k = 0; k < s0.size(); k++)
			{
				//Avoid adding actions that cannot change the model's state
				if(s0[k] == s1[k] && tp[k] == 1)continue;

				ite = find(S.begin(),S.end(),s0[k]);
				if(ite == S.end()) continue;
				ite = find(S.begin(),S.end(),s1[k]);
				if(ite == S.end()) continue;
				A.push_back(A_[gvi-1][j]);
				break;
			}
		}

		//-----------------------------
		//Z: observations that have non-prob of being perceived from
		// a state in S and an action in A
		for(unsigned int j = 0; j < A.size(); j++)
		{
			ite = find(A_[gvi-1].begin(),A_[gvi-1].end(),A[j]);
			int ai = distance(A_[gvi-1].begin(),ite);
			vector<string> s0 = get<0>(O_[gvi-1][ai]);
			vector<string> ob = get<1>(O_[gvi-1][ai]);
			for(unsigned int k = 0; k < s0.size(); k++)
			{
				ite = find(S.begin(),S.end(),s0[k]);
				if(ite != S.end())
				{
					ite = find(Z.begin(),Z.end(),ob[k]);
					if(ite == Z.end()) Z.push_back(ob[k]);
				}
			}
		}

		//-----------------------------
		//T & O: transition & observation probabilities for elments in S, Z & A
		for(unsigned int j = 0; j < A.size(); j++)
		{
			ite = find(A_[gvi-1].begin(),A_[gvi-1].end(),A[j]);
			int ai = distance(A_[gvi-1].begin(),ite);
			vector<string> s0 = get<0>(T_[gvi-1][ai]);
			vector<string> s1 = get<1>(T_[gvi-1][ai]);
			vector<double> tp = get<2>(T_[gvi-1][ai]);
			vector<string> ss = get<0>(O_[gvi-1][ai]);
			vector<string> ob = get<1>(O_[gvi-1][ai]);
			vector<double> op = get<2>(O_[gvi-1][ai]);

			//Get the T probabilities that start & end in states of S
			vector<string> tmp_s;
			vector<double> tmp_d;
			pMat tmp_T(tmp_s,tmp_s,tmp_d);
			for(unsigned int k = 0; k < s0.size(); k++)
			{
				ite = find(S.begin(),S.end(),s0[k]);
				if(ite != S.end())
				{
					//Evaluate if the ending state is in the local state space
					string loc_s1;
					ite = find(S.begin(),S.end(),s1[k]);
					if(ite != S.end()) loc_s1 = s1[k];
					else loc_s1 = "extra";

					//Check if the s0->"extra" transition has already been 
					bool not_repeated(true);
					if(loc_s1 == "extra")
					{
						for(unsigned int l = 0; l < get<0>(tmp_T).size(); l++)
						{
							if(get<0>(tmp_T)[l] == s0[k] && get<1>(tmp_T)[l] == "extra")
							{
								not_repeated = false;
								get<2>(tmp_T)[l] += tp[k];
								break;
							}
						}
					}

					//Insert the l-th transition probability
					if(not_repeated)
					{
						get<0>(tmp_T).push_back(s0[k]);
						get<1>(tmp_T).push_back(loc_s1);
						get<2>(tmp_T).push_back(tp[k]);
					}
				}
			}

			//Get the O probabilities that start & end in states of S
			pMat tmp_O(tmp_s,tmp_s,tmp_d);
			for(unsigned int k = 0; k < ss.size(); k++)
			{
				ite = find(S.begin(),S.end(),ss[k]);
				if(ite == S.end()) continue;

				//Save the l-th transition probability
				get<0>(tmp_O).push_back(ss[k]);
				get<1>(tmp_O).push_back(ob[k]);
				get<2>(tmp_O).push_back(op[k]);
			}

			//Add distributions for the extra state
			if(find(S.begin(),S.end(),"extra") != S.end())
			{
				get<0>(tmp_T).push_back("extra");
				get<1>(tmp_T).push_back("extra");
				get<2>(tmp_T).push_back(1.0);

				get<0>(tmp_O).push_back("extra");
				get<1>(tmp_O).push_back("none");
				get<2>(tmp_O).push_back(1.0);
			}

			//Save the j-th action's T-maxtrix
			T.push_back(tmp_T);
			//Save the j-th action's O-maxtrix
			O.push_back(tmp_O);
		}

		//-----------------------------
		//Terminate: add the elements associated to the "terminate" action
		S.push_back("absb-g");
		S.push_back("absb-ng");
		Z.push_back("none");
		A.push_back("terminate");
		vector<string> tmp_s;
		vector<double> tmp_d;
		pMat term_T(tmp_s,tmp_s,tmp_d);
		pMat term_O(tmp_s,tmp_s,tmp_d);
		//Set the T & O functions for the "terminate" action
		for(unsigned int j = 0; j < S.size(); j++)
		{
			//"terminate" transits from the goal state to "absb-g"
			if(S[j] == h_gs[i+1])
			{
				get<0>(term_T).push_back(S[j]);
				get<1>(term_T).push_back("absb-g");
				get<2>(term_T).push_back(1);
			}
			//"terminate" does not change absorbent states & "extra"
			else if(S[j] == "absb-g" || S[j] == "absb-ng" || S[j] == "extra")
			{
				get<0>(term_T).push_back(S[j]);
				get<1>(term_T).push_back(S[j]);
				get<2>(term_T).push_back(1);
			}
			//"terminate" moves any non-goal state to "absb-ng"
			else
			{
				get<0>(term_T).push_back(S[j]);
				get<1>(term_T).push_back("absb-ng");
				get<2>(term_T).push_back(1);
			}

			get<0>(term_O).push_back(S[j]);
			get<1>(term_O).push_back("none");
			get<2>(term_O).push_back(1);
		}
		//Save the T & O matrices for teh "terminate" action
		T.push_back(term_T);
		O.push_back(term_O);
		//Set the T & O probs. for absorbent states as starting state 
		//for every action different from "terminate"
		for(unsigned int j = 0; j < A.size(); j++)
		{
			if(A[j] == "terminate") continue;
			
			//Transition for "absb-g"
			get<0>(T[j]).push_back("absb-g");
			get<1>(T[j]).push_back("absb-g");
			get<2>(T[j]).push_back(1.0);
			get<0>(O[j]).push_back("absb-g");
			get<1>(O[j]).push_back("none");
			get<2>(O[j]).push_back(1.0);

			//Transition for "absb-ng"
			get<0>(T[j]).push_back("absb-ng");
			get<1>(T[j]).push_back("absb-ng");
			get<2>(T[j]).push_back(1.0);
			get<0>(O[j]).push_back("absb-ng");
			get<1>(O[j]).push_back("none");
			get<2>(O[j]).push_back(1.0);
		}

		//-----------------------------
		//help: add the elements associated to the "help" action
		if(find(S.begin(),S.end(),"extra") != S.end())
		{
			tmp_s.clear();
			tmp_d.clear();
			pMat help_T(tmp_s,tmp_s,tmp_d);
			pMat help_O(tmp_s,tmp_s,tmp_d);
			for(unsigned int j = 0; j < S.size(); j++)
			{
				if(S[j] == "absb-g" || S[j] == "absb-ng")
				{
					//"help" does not change absorbent states
					get<0>(help_T).push_back(S[j]);
					get<1>(help_T).push_back(S[j]);
					get<2>(help_T).push_back(1.0);
				}
				else
				{
					//"help" from any non-absorbent state goes to "absb-ng"
					get<0>(help_T).push_back(S[j]);
					get<1>(help_T).push_back("absb-ng");
					get<2>(help_T).push_back(1.0);
				}

				//Observation matrix
				get<0>(help_O).push_back(S[j]);
				get<1>(help_O).push_back("none");
				get<2>(help_O).push_back(1.0);
			}
			//Save the "help" action and its distributions
			A.push_back("help");
			T.push_back(help_T);
			O.push_back(help_O);
		}

		//Truncate probabilites precision to avoid precision errors
		//of complete prob. distributions
		for(unsigned int j = 0; j < A.size(); j++)
		{
			for(unsigned int k = 0; k < S.size(); k++)
			{
				//Truncate the precision of the k-th starting state T-distribution
				double total_p(0.0);
				double max_p(0.0);
				int max_i(-1);
				for(unsigned int l = 0; l < get<0>(T[j]).size(); l++)
				{
					if(get<0>(T[j])[l] == S[k])
					{
						get<2>(T[j])[l] = truncate(get<2>(T[j])[l],6);
						total_p += get<2>(T[j])[l];
						if(get<2>(T[j])[l] > max_p)
						{
							max_p = get<2>(T[j])[l];
							max_i = l;
						}
					}
				}
				double comp_p = 1 - total_p;
				get<2>(T[j])[max_i] += comp_p;

				//Truncate the precision of the k-th starting state T-distribution
				total_p = 0.0;
				max_p = 0.0;
				max_i = -1;
				for(unsigned int l = 0; l < get<0>(O[j]).size(); l++)
				{
					if(get<0>(O[j])[l] == S[k])
					{
						get<2>(O[j])[l] = truncate(get<2>(O[j])[l],6);
						total_p += get<2>(O[j])[l];
						if(get<2>(O[j])[l] > max_p)
						{
							max_p = get<2>(O[j])[l];
							max_i = l;
						}
					}
				}
				comp_p = 1 - total_p;
				get<2>(O[j])[max_i] += comp_p;
			}
		}

		//-----------------------------
		//R: define the reward & punishment for the local policy that has h_gs[i+1] as goal state
		string goal_state = h_gs[i+1];
		vector<tuple<string,string> > g_pair,p_pair;
		vector<string> p_vec;

		//For abstract actions, punish to  execute them from states that  are not their S0 by design
		if(gvi > 0)
		{
			for(unsigned int l = 0; l < A.size(); l++)
			{
				//Decompose AA's to get their intended S0 by design
				vector<string> a_vec = splitStr(A[l],"->");
				if(a_vec.size() != 2) continue;
				for(unsigned int m = 0; m < S.size(); m++)
				{
					//-1 will be the R only for the intended S0 or 'absb'
					if(a_vec[0] == S[m] || "absb-g" == S[m] || "absb-ng" == S[m] || "extra" == S[m]) continue;

					//Punish executing the l-th abstract action from non-intended S0 abstract states and 'extra'
					tuple<string,string> pp_tmp(S[m],A[l]);
					p_pair.push_back(pp_tmp);
				}
			}
		}

		//Reward signals associated to "extra" & "help"
		if(find(S.begin(),S.end(),"extra") != S.end())
		{
			//Punish executing any action diff. to "help" from "extra"
			for(unsigned int j = 0; j < A.size(); j++)
				if(A[j] != "help") p_pair.push_back(tuple<string,string>("extra",A[j]));

			//Punish executing "help" from any state diff. to "extra"
			for(unsigned int j = 0; j < S.size(); j++)
				if(S[j] != "extra") p_pair.push_back(tuple<string,string>(S[j],"help"));

			//Punish transiting to "extra"
			p_vec.push_back("extra");

			//Reward executing "help" from "extra"
			g_pair.push_back(tuple<string,string>("extra","help"));
		}

		//Reward signals associated to the "terminate" action
		for(unsigned int j = 0; j < S.size(); j++)
		{
			//Reward executing "terminate" from the goal state or "absb-g"
			if(S[j] == "absb-g" || S[j] == goal_state)
				g_pair.push_back(tuple<string,string>(S[j],string("terminate")));
			//Punish executing "terminate" from any state that is no the goal state or "absb-g"
			else p_pair.push_back(tuple<string,string>(S[j],string("terminate")));
		}

		//Verify that the POMDP is a valid one
		bool res = checkPomdp(S,A,Z,T,O);
		if(!res)
		{
			cout << ">> Error: invalid POMDP for local policy at level " << i+1 << "." << endl;
			return false;
		}

		//-----------------------------
		//POMDP: Generate the pomdp & compute the local policy
		bool success(false);
		double discount = 0.95;
		vector<string> r_order;//Reward printing order
		r_order.push_back("g");
		r_order.push_back("p");
		r_order.push_back("pp");
		r_order.push_back("gp");
		pomdp p = generatePomdp(S,A,Z,T,O,tmp_s,g_pair,p_vec,p_pair,discount,r_order,success);

		if(debug)
		{
			cout << "Goal state: " + goal_state << endl;
			cout << "Check pomdp: " << res << endl;
			cout << "Generate pomdp: " << success << endl;
		}

		//Solve the POMDP to get the policy
		policy loc_pol = solveModel(p,75,10,0.01);

		if(debug)
		{
			vector<string> pS = get<3>(loc_pol);
			vector<string> pA = get<1>(loc_pol);
			vector<string> pZ = get<2>(loc_pol);
			vector<string> pAS = get<4>(loc_pol);
			cout << ">> POLICY COMPONENTS"  << endl;
			cout << "S: \n\t";
			for(unsigned int j = 0; j < pS.size(); j++) cout << pS[j] + " ";
			cout << endl;
			cout << "A: \n\t";
			for(unsigned int j = 0; j < pA.size(); j++) cout << pA[j] + " ";
			cout << endl;
			cout << "Z: \n\t";
			for(unsigned int j = 0; j < pZ.size(); j++) cout << pZ[j] + " ";
			cout << endl;
			cout << "Abs. S: \n\t";
			for(unsigned int j = 0; j < pAS.size(); j++) cout << pAS[j] + " ";
			cout << endl;
			cout << "-------------------------------------------------" << endl;
		}

		//-----------------------------
		//Save the local policy
		h_pol.push_back(loc_pol);
	}

	//Store the hierarchical policy in the class' global variable
	HP_ = h_pol;

	return true;
}

bool HPomdp::hPolExec(int const &max_steps,bool const &debug)
{
	//----------------------------------------------------------------------------
	// EXECUTE H-POLICY
	if(HP_.size() == 0)
	{
		cout << ">> Error[hPolExec]: There is no hierarchical policy." << endl;
		return false;
	}

	//Start the execution of the hierarchical policy in a top-down manner
	for(int i = 0; i < HP_.size(); i++)
	{
		if(debug) cout << ">> START loc-pol(" << i << ")" << endl;

		//Execute the i-th local policy
		string last_a = execPA("",i,max_steps,debug);

		if(debug)
		{
			cout << ">> END loc-pol(" << i << "), last-action: " + last_a << endl;
			cout << "------------------------------------------" << endl;
		}

		//Determine the next policy to be executed
		if(last_a == "help") i -= 2;// (i-1)-th will be next
		else if(last_a == "terminate") continue;// (i+1)-th will be next
		else if(last_a.rfind("step", 0) == 0) return false;
		else if(last_a == "stuck") return false;
	}

	if(debug) cout << ">> Hierarchical policy done." << endl;

	return true;
}

string HPomdp::execPA(string const &a,int const &lp_id,int const &max_steps,bool const &debug)
{
	//----------------------------------------------------
	//Determine the case of execution
	int exe_case;
	vector<string>::iterator ite;
	map<string,policy>::iterator ite2 = AA_.find(a);
	if(ite2 != AA_.end())
	{
		//An action will be executed
		ite = find(A_[0].begin(),A_[0].end(),a);
		if(ite != A_[0].end())
		{
			//An concrete action will be executed
			exe_case = 0;
		}
		else
		{
			//An abstract action will be executed
			exe_case = 1;
		}
	}
	else
	{
		if(lp_id >= 0 && lp_id < HP_.size())
		{
			//A local policy will be executed
			exe_case = 2;
		}
		else
		{
			cout << ">> Error[execPA]: '" + a + "' is an invalid action and " << lp_id << " an invalid index for a local policy (HP_.size() is " << HP_.size() << ")." << endl;
			return string("");
		}
	}

	//----------------------------------------------------
	//Execute the concrete action if that is the case
	if(exe_case == 0)
	{
		string z = interactWithWorld(a);//Perform action
		updateBeliefDyn(a,z);//Update the global belief vector
		return z;
	}

	//----------------------------------------------------
	//Get the policy to be executed (local or AA)
	vector<string> tmp;
	policy pol(POMDP::Policy(1,1,1),tmp,tmp,tmp,tmp,POMDP::Model<MDP::Model>(1,1,1,0.9));//empty policy tuple
	if(exe_case == 1)//Get an abstract action
		pol = AA_.at(a);
	else if(exe_case == 2)//Get a local policy
		pol = HP_[lp_id];

	//----------------------------------------------------
	//Policy execution
	//Get the global-vector index at which the actions of the policy are
	int gvi(-1);
	int tdl(-1);
	for(unsigned int i = 0; i < hs_.depth(); i++)
	{
		vector<string> tmp = hs_.keysAtLevel(i);
		if(tmp.size() > 1)
		{
			tdl = i;
			break;
		}
	}
	if(exe_case == 1)
	{
		vector<string> pol_A = get<1>(pol);
		string test_a;
		for(unsigned int i = 0; i < pol_A.size(); i++)
		{
			if(pol_A[i] != "terminate")
			{
				test_a = pol_A[i];
				break;
			}
		}
		for(unsigned int i = 0; i < A_.size(); i++)
		{
			ite = find(A_[i].begin(),A_[i].end(),test_a);
			if(ite != A_[i].end())
			{
				gvi = i;
				break;
			}
		}
	}
	else if(exe_case == 2)
	{
		gvi = S_.size() - tdl - lp_id;
	}

	//Create the initial belief vector
	vector<string> pol_S = get<3>(pol);
	POMDP::Belief B(pol_S.size());
	ite = find(pol_S.begin(),pol_S.end(),"extra");
	int e_idx = distance(pol_S.begin(),ite);
	bool has_extra(ite != pol_S.end());
	for(unsigned int i = 0; i < pol_S.size(); i++) B(i) = 0.0;
	for(map<string,double>::iterator it = B_[gvi].begin(); it != B_[gvi].end(); ++it)
	{
		ite = find(pol_S.begin(),pol_S.end(),it->first);
		if(ite != pol_S.end())
		{
			//Assign prob to a known state
			int idx = distance(pol_S.begin(),ite);
			B(idx) = it->second;
		}
		//Unkown states' probs are added to the "extra" state
		else B(e_idx) += it->second;
	}

	//debug
	int os = (S_.size() - tdl) - gvi;
	string offset("");
	for(int i = 0; i < os; i++)
		offset += "\t";
	if(debug)
	{
		cout << offset + "INIT-B: ";
		for(unsigned int i = 0; i < pol_S.size(); i++)
			cout << pol_S[i] + "(" << B(i) << ") ";
		cout << endl;
	}

	//vector to monitor repeated abstract actions to detect if the system gets stuck
	vector<string> deb;

	//Compute the max-entropy for the extra state & get its index in the local S
	int extra_idx = e_idx;
	double max_se(0.0);
	double non_mod_s(S_[gvi].size());
	if(has_extra)
	{
		for(unsigned int i = 0; i < pol_S.size(); i++)
		{
			if(find(S_[gvi].begin(),S_[gvi].end(),pol_S[i]) != S_[gvi].end())
				non_mod_s -= 1.0;
		}
		max_se = log2(1/non_mod_s) * (1/non_mod_s);
		max_se *= (static_cast<double>(-1) * non_mod_s);
	}

	//Execution loop
	vector<string> pol_A = get<1>(pol);
	string last_action("");
	POMDP::Policy p_pol = get<0>(pol);
	POMDP::ValueFunction vf = p_pol.getValueFunction();
	int horizon = p_pol.getH();
	bool bl(false);
	while(true)
	{
		for(int t = horizon; t >= 1; t--)
		{
			//Sample the next action
			size_t action_idx;
			tuple<size_t,size_t> a_id;
			if(has_extra)
			{
				//The 'extra'-state's current entropy in the not-modeled region of S
				vector<double> se_vec;
				for(map<string,double>::iterator it = B_[gvi].begin(); it != B_[gvi].end(); ++it)
				{
					ite = find(pol_S.begin(),pol_S.end(),it->first);
					if(ite == pol_S.end() && it->second > 0.0) se_vec.push_back(it->second);
				}
				double se = entropy(se_vec);

				//The current entropy in the local state space
				se_vec.clear();
				for(unsigned int i = 0; i < pol_S.size(); i++)
				{
					if(pol_S[i] != "extra" && B(i) > 0.0)
						se_vec.push_back(B(i));
				}
				double loc_se = entropy(se_vec);

				//When there is an extra state, decide which selection criteria will be followed
				//based on the entropy in the modeled and un-modeled state space regions
				if(loc_se >= se)
				{
					//Get the action using the policy as interface
					a_id = p_pol.sampleAction(B,t);
					action_idx = get<0>(a_id);
				}
				else
				{
					//Get the action by accessing the value function
					action_idx = localSampleAction(vf,B,t,extra_idx,se,max_se);
				}
			}
			else
			{
				//Get the action using the policy as interface
				a_id = p_pol.sampleAction(B,t);
				action_idx = get<0>(a_id);
			}

			//Execute the action
			string a = pol_A[action_idx];
			string z;

			//debug
			if(gvi > 0)
			{
				//monitor only abstract actions
				if(deb.size() == 0) deb.push_back(a);
				else if(deb[deb.size()-1] == a) deb.push_back(a);
				else if(deb[deb.size()-1] != a)
				{
					deb.clear();
					deb.push_back(a);
				}

				//failure criteria
				if(deb.size() > 10) stuck = true;

				if(false)
				{
					//Write a file with info on the stuck policy
					ofstream outf("STUCK-POL.txt");
					outf << "S: ";
					for(unsigned int i = 0; i < pol_S.size(); i++) outf << pol_S[i]+" ";
					outf << endl;
					outf << "A: ";
					for(unsigned int i = 0; i < pol_A.size(); i++) outf << pol_A[i]+" ";
					outf << endl;

					for(unsigned int i = 0; i < pol_A.size(); i++)
					{
						int id = distance(A_[gvi].begin(),find(A_[gvi].begin(),A_[gvi].end(),pol_A[i]));
						outf << "T-DIST: "+pol_A[i] << endl;
						for(unsigned int j = 0; j < pol_S.size(); j++)
						{
							for(unsigned int k = 0; k < get<0>(T_[gvi][id]).size(); k++)
							{
							if(get<0>(T_[gvi][id])[k] == pol_S[j])
								outf << "\t"+get<0>(T_[gvi][id])[k]+"->"+get<1>(T_[gvi][id])[k]+": "<< get<2>(T_[gvi][id])[k] << endl;
							}
							outf << "\t-----------------------" << endl;
						}
					}
					outf.close();


					cout << endl << "GOT STUCK!" << endl;
					cin.get();
				}
			}

			//debug: display action
			if(debug)
			{
				if(gvi != 0)
				{
					cout << offset + a << endl;
					cin.get();
				}
				else if(a != "terminate" && a != "help")
					cout << offset + a + "-";
			}

			//Execute the action
			if(a == "terminate" || a == "help")
			{
				last_action = a;
				bl = true;
				break;
			}
			else z = execPA(a,-1,max_steps,debug);

			//Verify if the amount of steps taken so far are within the allowed range
			if(getStepCounter() > max_steps)
			{
				return string("step-limit-reached:") + to_string(max_steps);
			}

			//Verify if the agent got stuck
			if(stuck)
			{
				return string("stuck");
			}

			//debug: display observation for concrete actions
			if(debug)
			{
				if(gvi == 0)
				{
					cout << z << endl;
					cin.get();
				}
			}

			//Update the local belief vector
			for(unsigned int i = 0; i < pol_S.size(); i++) B(i) = 0.0;
			for(map<string,double>::iterator it = B_[gvi].begin(); it != B_[gvi].end(); ++it)
			{
				ite = find(pol_S.begin(),pol_S.end(),it->first);
				if(ite != pol_S.end())
				{
					//Assign prob to a known state
					int idx = distance(pol_S.begin(),ite);
					B(idx) = it->second;
				}
				//Unkown states' probs are added to the "extra" state
				else B(e_idx) += it->second;
			}
		}

		if(bl) break;
	}

	//debug
	if(debug)
	{
		cout << offset + "END-B: ";
		for(unsigned int i = 0; i < pol_S.size(); i++)
			cout << pol_S[i] + "(" << B(i) << ") ";
		cout << endl;
	}

	//----------------------------------------------------
	//Return final action of the policy ("terminate" or "help")
	return last_action;
}

double HPomdp::entropy(vector<double> const &p)
{
	//Make sure the probabilities are normalized
	double total(0.0);
	vector<double> norm_p;
	for(unsigned int i = 0; i < p.size(); i++)
	{
		total += p[i];
if(isnan(p[i]))
{
cout << endl << "[entropy]:" << i << "-th of p is nan." << endl;
cin.get();
}
	}

if(isnan(total))
{
cout << endl << "[entropy]: total is nan." << endl;
cin.get();
}

	if(total == static_cast<double>(0.0)) return 0.0;

	for(unsigned int i = 0; i < p.size(); i++)
	{
		norm_p.push_back(p[i]/total);
if(isnan(p[i]/total))
{
cout << endl << "[entropy][norm_p]:" << p[i] << "/" << total << " is nan." << endl;
cin.get();
}
	}

	//Compute the Shannon entropy
	double sh_e(0.0);
	for(unsigned int i = 0; i < norm_p.size(); i++)
	{
		sh_e -= (norm_p[i] * log2(norm_p[i]));
bool tmp(false);
if(isnan(norm_p[i])){cout << endl << "[entropy][sh_e]np: is nan." << endl; tmp = true;}
if(isnan(log2(norm_p[i]))){cout << endl << "[entropy][sh_e]lg2(np):log2 of "<< norm_p[i] << " is nan." << endl; tmp = true;}
if(isnan(norm_p[i] * log2(norm_p[i]))){cout << endl << "[entropy][sh_e]np*lg2(np):" << norm_p[i] << "*" << log2(norm_p[i]) << " is nan." << endl; tmp = true;}
if(tmp) cin.get();
	}

	return sh_e;
}

double HPomdp::dotP(POMDP::Belief const &b, Eigen::Matrix<double,Eigen::Dynamic,1> const &a)
{
	double dp(0.0);
	for(unsigned int i = 0; i < a.rows(); i++)
		dp += (b(i) * a(i,0));
	return dp;
}

size_t HPomdp::localSampleAction(POMDP::ValueFunction const &vf,POMDP::Belief const &b,int const &t,int const &extra_idx,double const &se,double const &max_se)
{
	//Sample for the 't'-remaining-steps step
	double b_val(0.0);
	int idx(-1);
	for(unsigned int i = 0; i < vf[t].size(); i++)
	{
		//Get the i-th alpha vector
		MDP::Values alpha = vf[t][i].values;

		//Weight the value of the 'extra' state based on its current entropy
		alpha(extra_idx,0) = alpha(extra_idx,0) / (1 + abs(alpha(extra_idx,0)*(se / max_se))); //eq.1

		//Compute the belief's value for the i-th alpha vector
		double i_val = dotP(b,alpha);
		if(i == 0 || i_val > b_val)
		{
			b_val = i_val;
			idx = i;
		}
	}

	//Return the index of the action whose alpha vector got the
	//highest value for belief 'b'
	return vf[t][idx].action;
}

bool HPomdp::run(int argc,char** argv)
{
	if(argc < 8)
	{
		//Parameters used in experiments reported in the forst commit of the 'ThesisExperiments' repo
		cout << "Usage: ./debug <# buildings> <std-dev> <building-dim> <room-dim> <subsection-dim> <# run sim.> <results dir.>" << endl;
		return false;
	}

	//Get experimental configuration parameters
	unsigned n_building	= stoi(string(argv[1]));
	double std_dev		= stod(string(argv[2]));
	unsigned build_dim	= stoi(string(argv[3]));
	unsigned room_dim	= stoi(string(argv[4]));
	unsigned subsec_dim	= stoi(string(argv[5]));
	unsigned n_run		= stoi(string(argv[6]));
	string res_dir		= string(argv[7]);
	string control_var	= "--";
	double control_val	= 0.0;
	unsigned kernel_dim	= 3;
	double room_conn_ratio	= 0.5;
	unsigned b_dim		= build_dim;
	unsigned r_dim		= room_dim;
	unsigned s_dim		= subsec_dim;
	unsigned h_dim_w	= 2;
	unsigned h_dim_h	= 1;
	unsigned hall_flag	= EnvGen::DONT_USE_HALLS;
	double T_prec		= 0.90;
	int n_sim_aa		= 100;

	//----------------------------------------------------------------------------------
	//----------------------------------------------------------------------------------
	//INITIAL EXPERIMENT SETTING

	//Generate the output file directory and create it
	string hp_dir(res_dir+"/H-POMDP");
	string tp_dir(res_dir+"/TwoL-POMDP");
	string fp_dir(res_dir+"/Flat-POMDP");
	shellCmd("mkdir "+res_dir);
	shellCmd("mkdir "+hp_dir);
	shellCmd("mkdir "+tp_dir);
	shellCmd("mkdir "+fp_dir);

	//Generate the evaluation environment
	json egj;
	egj["n_b"] = n_building;
	egj["hall_flag"] = hall_flag;
	egj["use_hall_prob"] = 0.5;
	egj["room_conn_ratio"] = room_conn_ratio;
	egj["cell_pixel_dim"] = 40;
	egj["b_dims"] = json::array();
	egj["h_dims"] = json::array();
	egj["r_dims"] = json::array();
	egj["s_dims"] = json::array();
	for(unsigned int i = 0; i < 4; i++)
	{
		if(i < 2) egj["h_dims"].push_back(h_dim_w);
		else egj["h_dims"].push_back(h_dim_h);
		egj["b_dims"].push_back(b_dim);
		egj["r_dims"].push_back(r_dim);
		egj["s_dims"].push_back(s_dim);
	}
	ofstream file_writer(res_dir+"/config-env.json");
	file_writer << egj.dump(4);
	file_writer.close();
	EnvGen eg(res_dir+"/config-env.json");
	eg.generateEnv(res_dir+"/env.json",res_dir+"/env.jpg");

	//Create the concrete probabilities file
	double O_prec(0.10);
	createProbFile(res_dir+"/prob.json",T_prec,O_prec,kernel_dim,std_dev);

	//Load the environment & initialize the simulator
	std::chrono::high_resolution_clock::time_point tmp_start = std::chrono::high_resolution_clock::now();
	HPomdp hp(res_dir+"/env.json",res_dir+"/prob.json",n_sim_aa);
	std::chrono::high_resolution_clock::time_point tmp_end = std::chrono::high_resolution_clock::now();
	hp.setSimulatorModel();

	//Initial planning time for H-POMDP
	double init_p_time = u2s(tmp_start,tmp_end);

	//Load the environment in separate TH & NH to determine sub-region paths
	TreeHandle th;
	Neighborhood nh;
	bool r1 = th.navFromJson(res_dir+"/env.json");
	bool r2 = nh.navFromJson(res_dir+"/env.json");
	nh.propNeigNav(th);

	//Generate the flat POMDP
	pomdp p = hp.concModel();

	//Get the list of cells from the extreme buildings for stratified sampling
	vector<string> tmp_bui = th.keysAtLevel(1);
	vector<string> bc1,bc2,both_bc;
	for(unsigned int i = 0; i < tmp_bui.size(); i++)
	{
		bool res;
		vector<string> tmp_n = nh.neigTo(tmp_bui[i],res);
		if(tmp_n.size() == 1)
		{
			//Gather the cells of this extreme building
			vector<string> tmp_r,tmp_c;
			tmp_r = th.keysOfChildren(tmp_bui[i]);
			for(unsigned int j = 0; j < tmp_r.size(); j++)
			{
				vector<string> tmp_s = th.keysOfChildren(tmp_r[j]);
				for(unsigned int k = 0; k < tmp_s.size(); k++)
				{
					vector<string> tmp_cc = th.keysOfChildren(tmp_s[k]);
					tmp_c.insert(tmp_c.end(),tmp_cc.begin(),tmp_cc.end());
				}
			}

			//Save the cells
			if(bc1.size() == 0) bc1 = tmp_c;
			else bc2 = tmp_c;
		}
	}
	both_bc = bc1;
	both_bc.insert(both_bc.end(),bc2.begin(),bc2.end());

	//Generate the pairs of init-goal states & their shortest path length
	vector<string> init_s,goal_s;
	vector<double> ig_len;
	cout << "--------------------------------------" << endl;
	cout << "GENERATING " << n_run << " INIT-GOAL PAIRS: ";
	for(unsigned int i = 0; i < n_run; i++)
	{
		//The init-goal pairs are sampled in a stratified manner:
		// - If the environment has only 1 building, random sampling is performed.
		// - If there are at least 2 buildings, the init and goal states will belong to 
		//   opposite extreme buildings. 
		string r_s0;
		string r_gs;
		if(tmp_bui.size() >= 2)
		{
			r_s0 = both_bc[rand() % both_bc.size()];
			auto ite = find(bc1.begin(),bc1.end(),r_s0);
			if(ite == bc1.end()) r_gs = bc1[rand() % bc1.size()];
			else r_gs = bc2[rand() % bc2.size()];
		}
		else
		{
			while(true)
			{
				r_s0 = get<1>(p)[rand() % get<1>(p).size()];
				r_gs = get<1>(p)[rand() % get<1>(p).size()];
				if(r_s0 != r_gs && r_s0 != "absb" && r_gs != "absb") break;
			}
		}

		double opt_d = static_cast<double>(th.optPath(nh,r_s0,r_gs,false));

		//Display that this pair has been created
		if(i > 0) cout << ",";
		cout << i+1;
		cout.flush();

		//Save the init-goal pair & its length
		init_s.push_back(r_s0);
		goal_s.push_back(r_gs);
		ig_len.push_back(opt_d);
	}
	cout << endl;

	//----------------------------------------------------------------------------------
	//----------------------------------------------------------------------------------
	//RUN SIMULATIONS
	cout << "--------------------------------------" << endl;
	cout << "# OF RUNS ON EACH MODEL: " << init_s.size() << endl;
	cout << "--------------------------------------" << endl;

	//debug flag
	bool debug(false);

	//Vectors for storing the 'n_run' results
	vector<double> success_vec;
	vector<double> steps_vec;
	vector<double> ratio_vec;
	vector<double> rel_error_vec;
	vector<double> error_vec;
	vector<double> time_vec;

	//++++++++++++++++++++++++++++++++++++++++++++++++
	//Hierarchical POMDP

	//Create file to save the results of each run
	ofstream hp_res_file;
	hp_res_file.open(hp_dir+"/sim_res.csv");
	hp_res_file << "success,steps taken,path relative cost,relative error,absolute error,planning time,";
	hp_res_file << "initial state,goal state,final state,shortest path length" << endl;
	hp_res_file.close();

	//Run
	cout << "H-POMDP: ";
	cout.flush();
	int stuck_count(0);
	for(unsigned int i = 0; i < init_s.size(); i++)
	{
		//Reset the simulator
		hp.resetStepCounter();
		hp.setSimulatorState(init_s[i]);

		//Uncomment the line below so that the HP-agent starts its  task with a uniform belief distribution
		//hp.uncertainS();

		//Compute the hierarchical policy
		tmp_start = std::chrono::high_resolution_clock::now();
		bool p_status = hp.hPolPlan(goal_s[i],debug);
		tmp_end = std::chrono::high_resolution_clock::now();

		//Max amount of allowed concrete steps
		int max_steps = hp.getSizeS(0);

		//Execute the hierarchical policy
		bool e_status = hp.hPolExec(max_steps,debug);

		//Generate the i-th run result
		string f_state = hp.getSimulatorState();
		double f_success = ((f_state == goal_s[i]) ? 1.0 : 0.0);
		double f_steps = hp.getStepCounter();
		double f_ratio = f_steps / ig_len[i];
		double f_error = th.optPath(nh,f_state,goal_s[i],false);
		double p_time = u2s(tmp_start,tmp_end);
		double rel_error = f_error / ig_len[i];

		//Store results in vectors
		success_vec.push_back(f_success);
		steps_vec.push_back(f_steps);
		ratio_vec.push_back(f_ratio);
		error_vec.push_back(f_error);
		time_vec.push_back(p_time);
		rel_error_vec.push_back(rel_error);

		//Save i-th run result in file
		//success-steps-optratio-relerror-manerror-ptime-s_0-s_g-s_f-iglen
		fstream fs;
		fs.open(hp_dir+"/sim_res.csv",std::fstream::app);
		fs << f_success << "," << f_steps << "," << f_ratio << "," << rel_error;
		fs << "," << f_error << "," << p_time << "," << init_s[i] << "," << goal_s[i] << "," << f_state;
		fs << "," << ig_len[i] << endl;
		fs.close();

		if(i > 0) cout << ",";
		cout << i+1;
		cout.flush();
	}
	cout << endl;

	//Save summarize file
	saveSummFile(success_vec,
		     steps_vec,
		     ratio_vec,
		     error_vec,
		     time_vec,
		     rel_error_vec,
		     ig_len,
		     control_var,
		     control_val,
		     hp_dir+"/summarize_results.txt"
		     );

	//Add the initial planning time for Hie-POMDP
	fstream hp_app;
	hp_app.open(hp_dir+"/summarize_results.txt",std::fstream::app);
	hp_app << "INIT PLANNING TIME:" << endl;
	hp_app << init_p_time << endl;
	hp_app.close();

	//++++++++++++++++++++++++++++++++++++++++++++++++
	//Two-level POMDP
	success_vec.clear();
	steps_vec.clear();
	ratio_vec.clear();
	error_vec.clear();
	time_vec.clear();
	rel_error_vec.clear();

	//Create file to save the results of each run
	ofstream tp_res_file;
	tp_res_file.open(tp_dir+"/sim_res.csv");
	tp_res_file << "success,steps taken,path relative cost,relative error,absolute error,planning time,";
	tp_res_file << "initial state,goal state,final state,shortest path length" << endl;
	tp_res_file.close();

	//Initial planning
	//Create the list of policies to transit between
	tmp_start = std::chrono::high_resolution_clock::now();
	map<string,policy> pols = hp.srPolicies();
	tmp_end = std::chrono::high_resolution_clock::now();
	init_p_time = u2s(tmp_start,tmp_end);

	//Get the list of buildings
	vector<string> sr = th.keysAtLevel(1);

	//Run
	cout << "2-lvl-POMDP: ";
	cout.flush();
	for(unsigned int i = success_vec.size(); i < init_s.size(); i++)
	{
		//Reset the simulator
		hp.resetStepCounter();
		hp.setSimulatorState(init_s[i]);

		//Determine sub-regions of start & goal state
		tmp_start = std::chrono::high_resolution_clock::now();
		string sr_0,sr_g;
		for(unsigned int j = 0; j < sr.size(); j++)
		{
			if(th.isAncestor(sr[j],init_s[i])) sr_0 = sr[j];
			if(th.isAncestor(sr[j],goal_s[i])) sr_g = sr[j];
		}

		//Compute policy for final sub-region
		policy f_pol = hp.finalPol(sr_g,goal_s[i]);

		//Select sequence of policies
		vector<string> seq,path;
		if(sr_0 != sr_g)//to be tested
		{
			path.push_back(sr_0);
			nh.recPath(path,sr_g);
			for(unsigned int j = 0; j < (path.size()-1); j++)
			{
				seq.push_back(path[j]+path[j+1]);
			}
			path.erase(path.begin());
		}
		tmp_end = std::chrono::high_resolution_clock::now();

		//Execute sub-region policies
		string mon_f("");
		if(sr_0 != sr_g)
		{
			for(unsigned int j = 0; j < seq.size(); j++)
			{
				policy loc_pol = pols.at(seq[j]);
				int max_s = get<3>(loc_pol).size();//half #S is max-steps allowed to be taken
				string init;
				if(j == 0) init = init_s[i];
				else
				{
					vector<string>::iterator ite;
					ite = find(get<3>(loc_pol).begin(),get<3>(loc_pol).end(),mls);
					if(ite != get<3>(loc_pol).end() && mls != "terminate") init = mls;
					else init = "not-state";
				}
				tuple<string,int> res = hp.exePol(loc_pol,max_s,init);

				//Reaching max-step or stopping in a non-goal sub-region is a fail run
				if(get<0>(res) != "terminate" || !th.isAncestor(path[j],hp.getSimulatorState()))
				{
					mon_f = "fail";
					break;
				}
			}
		}

		//Execute final policy only if the goal-building was reached
		if(mon_f != "fail")
		{
			int max_s = get<3>(f_pol).size();//half #S is max-steps allowed to be taken
			tuple<string,int> res = hp.exePol(f_pol,max_s);
		}

		//Generate the i-th run result
		string f_state = hp.getSimulatorState();
		double f_success = ((f_state == goal_s[i]) ? 1.0 : 0.0);
		double f_steps = hp.getStepCounter();
		double f_ratio = f_steps / ig_len[i];
		double f_error = th.optPath(nh,f_state,goal_s[i],false);
		double p_time = u2s(tmp_start,tmp_end);
		double rel_error = f_error / ig_len[i];

		//Store results in vectors
		success_vec.push_back(f_success);
		steps_vec.push_back(f_steps);
		ratio_vec.push_back(f_ratio);
		error_vec.push_back(f_error);
		time_vec.push_back(p_time);
		rel_error_vec.push_back(rel_error);

		//Save i-th run result in file
		//success-steps-optratio-relerror-manerror-ptime-s_0-s_g-s_f-iglen
		fstream fs;
		fs.open(tp_dir+"/sim_res.csv",std::fstream::app);
		fs << f_success << "," << f_steps << "," << f_ratio << "," << rel_error;
		fs << "," << f_error << "," << p_time << "," << init_s[i] << "," << goal_s[i] << "," << f_state;
		fs << "," << ig_len[i] << endl;
		fs.close();

		if(i > 0) cout << ",";
		cout << i+1;
		cout.flush();
	}
	cout << endl;

	//Save summarize file
	saveSummFile(success_vec,
		     steps_vec,
		     ratio_vec,
		     error_vec,
		     time_vec,
		     rel_error_vec,
		     ig_len,
		     control_var,
		     control_val,
		     tp_dir+"/summarize_results.txt"
		     );

	//Add the initial planning time for Hie-POMDP
	fstream tp_app;
	tp_app.open(tp_dir+"/summarize_results.txt",std::fstream::app);
	tp_app << "INIT PLANNING TIME:" << endl;
	tp_app << init_p_time << endl;
	tp_app.close();

	//++++++++++++++++++++++++++++++++++++++++++++++++
	//Flat POMDP
	success_vec.clear();
	steps_vec.clear();
	ratio_vec.clear();
	error_vec.clear();
	time_vec.clear();
	rel_error_vec.clear();

	//Create file to save the results of each run
	ofstream fp_res_file;
	fp_res_file.open(fp_dir+"/sim_res.csv");
	fp_res_file << "success,steps taken,path relative cost,relative error,absolute error,planning time,";
	fp_res_file << "initial state,goal state,final state,shortest path length" << endl;
	fp_res_file.close();

	//Run
	cout << "Flat-POMDP: ";
	cout.flush();
	for(unsigned int i = success_vec.size(); i < init_s.size(); i++)
	{
		//Reset the simulator
		hp.resetStepCounter();
		hp.setSimulatorState(init_s[i]);

		//Compute policy for given goal state
		vector<string> G,P;
		vector< tuple<string,string> > Gpair,Ppair;
		Ppair.push_back(tuple<string,string>("*","terminate"));
		Gpair.push_back(tuple<string,string>(goal_s[i],"terminate"));
		bool success;
		double discount = 0.95;
		vector<string> order;
		order.push_back("g");
		order.push_back("p");
		order.push_back("pp");
		order.push_back("gp");
		pomdp model = generatePomdp(get<1>(p),get<2>(p),get<3>(p),get<4>(p),get<5>(p),G,Gpair,P,Ppair,discount,order,success);

		//Compute the flat-POMDP policy
		tmp_start = std::chrono::high_resolution_clock::now();
		policy pol = solveModel(model,200,10,0.1);
		tmp_end = std::chrono::high_resolution_clock::now();

		//start execution
		int max_s = get<1>(p).size();
		tuple<string,int> res = hp.exePol(pol,max_s,init_s[i]);

		//Generate the i-th run result
		string f_state = hp.getSimulatorState();
		double f_success = ((f_state == goal_s[i]) ? 1.0 : 0.0);
		double f_steps = hp.getStepCounter();
		double f_ratio = f_steps / ig_len[i];
		double f_error = th.optPath(nh,f_state,goal_s[i],false);
		double p_time = u2s(tmp_start,tmp_end);
		double rel_error = f_error / ig_len[i];

		//Store results in vectors
		success_vec.push_back(f_success);
		steps_vec.push_back(f_steps);
		ratio_vec.push_back(f_ratio);
		error_vec.push_back(f_error);
		time_vec.push_back(p_time);
		rel_error_vec.push_back(rel_error);

		//Save i-th run result in file
		fstream fs;
		fs.open(fp_dir+"/sim_res.csv",std::fstream::app);
		fs << f_success << "," << f_steps << "," << f_ratio << "," << rel_error;
		fs << "," << f_error << "," << p_time << "," << init_s[i] << "," << goal_s[i] << "," << f_state;
		fs << "," << ig_len[i] << endl;
		fs.close();

		if(i > 0) cout << ",";
		cout << i+1;
		cout.flush();
	}
	cout << endl;

	//Save summarize file
	saveSummFile(success_vec,
		     steps_vec,
		     ratio_vec,
		     error_vec,
		     time_vec,
		     rel_error_vec,
		     ig_len,
		     control_var,
		     control_val,
		     fp_dir+"/summarize_results.txt"
		     );
	return true;
}

//----------------------------------------------------------------------------------------------------------
//Utility methods

bool HPomdp::createProbFile(string const &file,double const &T_prec, double const &O_prec, int O_kernel_size, double sd)
{
	//Check for valid parameters
	ofstream tmpfile(file);
	if(!tmpfile.is_open()) return false;
	else tmpfile.close();
	if(T_prec < 0 || T_prec > 1) return false;
	if(O_prec < 0 || O_prec > 1) return false;

	//Write the T & O functions default probabilities
	json jj;
	jj["Function"] =
	{
		{
			{"type","T"},
			{"up",T_prec},
			{"down",T_prec},
			{"left",T_prec},
			{"right",T_prec}
		},
		{
			{"type","O"},
			{"precision",O_prec}
		}
	};

	//Check if a gaussian kernel will created
	if(O_kernel_size >= 3 && sd >= 0)
	{
		//Check that the kernal dimension is an odd number
		if(O_kernel_size % 2 == 1)
		{
			//Create a NxN empty kernel
			vector< vector<double> > kernel(O_kernel_size,vector<double>(O_kernel_size,0.0));

			double sum(0.0);//Sum for normalization
			double sigma = sd;//std. dev. 
			double s = 2.0 * sigma * sigma; 
			int lmt = floor(O_kernel_size / 2);
			for(int i = -lmt; i <= lmt ; i++)
			{
				for(int j = -lmt; j <= lmt ; j++)
				{
					double r = sqrt(i * i + j * j);
					kernel[i + lmt][j + lmt] = (exp(-(r * r) / s)) / (M_PI * s);
					sum += kernel[i + lmt][j + lmt];
				}
			}

			//Perform normalization
			for(unsigned int i = 0; i < kernel.size(); i++)
			{
				for(unsigned int j = 0; j < kernel[i].size(); j++)
				{
					kernel[i][j] /= sum;
				}
			}

			//Save the Shannon entropy for this O-distribution
			double entropy(0.0);
			for(unsigned int i = 0; i < kernel.size(); i++)
			{
				for(unsigned int j = 0; j < kernel[i].size(); j++)
				{
					entropy -= (kernel[i][j] * log2(kernel[i][j]));
				}
			}
			vector<string> tmpv = splitStr(file,"/");
			string iov("");
			if(file[0] == '/') iov += "/";
			for(unsigned int i = 0; i < tmpv.size()-1; i++)
			{
				iov += tmpv[i];
				iov += "/";
			}
			iov += "ind-ope-var.txt";

			ofstream iov_file(iov);
			iov_file << "Shannon entropy in obs. dist." << endl;
			iov_file << entropy << endl;
			iov_file.close();


			//Append the kernel elements to the JSON structure
			jj["Function"].push_back(
			{
				{"type","O-dist"},
				{"p-dist",
					{
						//Append the central prob
						{
							{"x",0},
							{"y",0},
							{"prob", kernel[lmt][lmt]}
						}
					}
				}
			});

			//Get the index of the function that will hold the gaussian dist.
			int idx(-1);
			for(unsigned int i = 0; i < jj["Function"].size(); i++)
			{
				if(jj["Function"][i]["type"] == "O-dist")
				{
					idx = i;
					break;
				}
			}

			for(int i = 0; i < kernel.size(); i++)
			{
				for(int j = 0; j < kernel[i].size(); j++)
				{
					//Do not append the central element again
					if(i == lmt && j == lmt) continue;

					jj["Function"][idx]["p-dist"].push_back(
					{
						{"x",j-lmt},
						{"y",i-lmt},
						{"prob",kernel[i][j]}
					});
				}
			}
		}
	}

	//Save the JSON file
	ofstream outfile(file);
	outfile << jj.dump(4);
	outfile.close();

	return true;
}

double HPomdp::truncate(double const &d,unsigned const &dec_prec)
{
	//Convert the variable into a string in order to evaluate in which
	//case it is
	stringstream ss;
	ss << std::setprecision(15);
	ss << d;
	string sd = ss.str();

	//For values expressed in scientific notation
	if(sd.find("e") != -1)
	{
		string sub_sd = sd.substr(sd.find("e")+2);
		int n_zeros = stoi(sub_sd)-1;

		//This first non-zero digit is beyond the requested truncate precision
		if(n_zeros >= static_cast<int>(dec_prec)) return 0.0;
		//The truncated value is different from zero
		else
		{
			string t_sd;
			if(d  < 0) t_sd = "-0.";
			else t_sd = "0.";
			for(int i = 0; i < n_zeros; i++) t_sd.push_back('0');

			int dig_c(0);
			for(int i = 0; i < sd.length(); i++)
			{
				if(dig_c == (static_cast<int>(dec_prec) - n_zeros)) break;

				if(sd[i] == 'e') break;
				else if(sd[i] == '-' || sd[i] == '.') continue;
				else
				{
					t_sd.push_back(sd[i]);
					dig_c++;
				}
			}

			return stod(t_sd);
		}
	}
	//For values that are not expressed in scientific notation
	else
	{
		bool passed_p(false);
		string t_sd("");
		int cnt(0);
		for(int i = 0; i < sd.length(); i++)
		{
			t_sd.push_back(sd[i]);

			if(sd[i] == '.')
			{
				passed_p = true;
				continue;
			}

			if(passed_p) cnt++;

			if(cnt == static_cast<int>(dec_prec)) break;
		}

		//Convert the resulting string to double
		return stod(t_sd);
	}
}

std::string HPomdp::shellCmd(std::string cmd)
{
	std::string data;
	FILE * stream;
	const int max_buffer = 256;
	char buffer[max_buffer];
	cmd.append(" 2>&1");

	stream = popen(cmd.c_str(), "r");
	if(stream)
	{
		while(!feof(stream))
		if(fgets(buffer,max_buffer,stream) != nullptr) data.append(buffer);
		pclose(stream);
	}
	return data;
}

double HPomdp::u2s(std::chrono::time_point<std::chrono::high_resolution_clock> t1,std::chrono::time_point<std::chrono::high_resolution_clock> t2)
{
	unsigned int elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>
				 (t2 - t1).count();
	double elapsed_s = static_cast<double>(elapsed_us) / 1000000.0;

	//cout << "micro: " << elapsed_us << endl;
	//cout << "sec: " << elapsed_s << endl;

	return elapsed_s;
}

tuple<double,double> HPomdp::avgSd(vector<double> const &vec)
{
	double avg(0.0);
	for(unsigned int i = 0; i < vec.size(); i++) avg += vec[i];
	avg /= static_cast<double>(vec.size());

	double sd(0.0);
	for(unsigned int i = 0; i < vec.size(); i++)
	{
		double tmp = (vec[i]-avg)*(vec[i]-avg);
		sd += tmp;
	}
	sd /= static_cast<double>(vec.size());
	sd = sqrt(sd);

	tuple<double,double> res(avg,sd);
	return res;
}

vector<std::string> HPomdp::splitStr(std::string const &fullString,std::string const &delimiter)
{
	//Verify that the delimiter is not an empty string
	int dlen = delimiter.length();
	if(dlen <= 0) return std::vector<std::string>();

	std::vector<std::string> tokens;

	//The delimiter does not exist in fullString
	if(fullString.find(delimiter) == -1)
	{
		tokens.push_back(fullString);

		return tokens;
	}

	std::string temp = fullString;
	while(true)
	{
		int index = temp.find(delimiter);
		std::string token;

		//The delimiter does not exist in 'temp'
		if(index == -1)
		{
			//The last token that doesn't end in the 'delimiter'
			token = temp;
			if(token != "") tokens.push_back(token);
			break;
		}
		else if(index == temp.length()-dlen)
		{
			//The index is the last character, therefore, this is the last iteration
			token = temp.substr(0,index);
			if(token != "") tokens.push_back(token);
			break;
		}
		else
		{
			token = temp.substr(0,index);
			if(token != "") tokens.push_back(token);
			temp = temp.substr(index+dlen,-1);
		}
	}

	return tokens;
}

bool HPomdp::saveSummFile(vector<double> const &v1,vector<double> const &v2,vector<double> const &v3,vector<double> const &v4,vector<double> const &v5,vector<double> const &v6,vector<double> const &v7,string const &var_name,double const &var_value,string const &f_name)
{
	ofstream outf;
	outf.open(f_name);
	if(!outf.is_open()) return false;

	//Filter the opt-rate to consider only the runs in which the goal state was reached
	vector<double> filter_v3;
	for(unsigned int i = 0; i < v1.size(); i++)
	{
		if(v1[i] == 1) filter_v3.push_back(v3[i]);
	}

	//Compute the average & std-dev of each vector
	tuple<double,double> tmp1 = avgSd(v1);
	tuple<double,double> tmp2 = avgSd(v2);
	tuple<double,double> tmp3 = avgSd(filter_v3);
	tuple<double,double> tmp4 = avgSd(v4);
	tuple<double,double> tmp5 = avgSd(v5);
	tuple<double,double> tmp6 = avgSd(v6);
	tuple<double,double> tmp7 = avgSd(v7);

	outf << "Control var name:" << endl;
	outf << var_name << endl;
	outf << "Control var value:" << endl;
	outf << var_value << endl;

	outf << "SUCCESS RATIO:" << endl;
	outf << get<0>(tmp1) << endl;
	outf << "# SUCCESSFUL RUNs:" << endl;
	outf << filter_v3.size() << endl;

	outf << "Avg. # STEPS:" << endl;
	outf << get<0>(tmp2) << endl;
	outf << "Std-dev:" << endl;
	outf << get<1>(tmp2) << endl;

	outf << "Avg. OPT-RATIO:" << endl;
	outf << get<0>(tmp3) << endl;
	outf << "Std-dev:" << endl;
	outf << get<1>(tmp3) << endl;

	outf << "Avg. MANH-ERROR:" << endl;
	outf << get<0>(tmp4) << endl;
	outf << "Std-dev:" << endl;
	outf << get<1>(tmp4) << endl;

	outf << "Avg. PLAN-TIME:" << endl;
	outf << get<0>(tmp5) << endl;
	outf << "Std-dev:" << endl;
	outf << get<1>(tmp5) << endl;

	outf << "Avg. RELATIVE ERROR:" << endl;
	outf << get<0>(tmp6) << endl;
	outf << "Std-dev:" << endl;
	outf << get<1>(tmp6) << endl;

	outf << "Avg. AVERAGE OPTIMUM PLAN LENGTH:" << endl;
	outf << get<0>(tmp7) << endl;
	outf << "Std-dev:" << endl;
	outf << get<1>(tmp7) << endl;

	outf.close();

	return true;
}

