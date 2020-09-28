#include <Neighborhood.hpp>

using namespace std;
using json = nlohmann::json;

string Neighborhood::stripQuotes(string const &s)
{
	string out_s("");
	for(int i = 0; i < s.length(); i++)
	{
		if(s[i] != '\"') out_s += s[i];
	}

	return out_s;
}

Neighborhood::Neighborhood()
{

}

bool Neighborhood::navFromJson(string file_json)
{
	//Check for a valid file
	ifstream infile(file_json);
	if(!infile.is_open())
	{
		cout << "ERROR[navFromJson]: could not open " + file_json << endl;
		return false;
	}

	try
	{
		//Clean the current vectors
		states.clear();
		neig_to.clear();
		to_left_of_.clear();
		to_right_of_.clear();
		above_.clear();
		below_.clear();

		//Parse the file into a JSON structure
		json jj;
		infile >> jj;
		infile.close();

		//Gather all the states
		vector<string>::iterator ite;
		for(unsigned int i = 0; i < jj["to_left_of"].size(); i++)
		{
			string tmp = stripQuotes(jj["to_left_of"][i]["subject"]);
			ite = find(states.begin(),states.end(),tmp);
			if(ite == states.end())
			{
				//Add a new state
				states.push_back(tmp);
			}
			tmp = stripQuotes(jj["to_left_of"][i]["reference"]);
			ite = find(states.begin(),states.end(),tmp);
			if(ite == states.end())
			{
				//Add a new state
				states.push_back(tmp);
			}
		}
		for(unsigned int i = 0; i < jj["above"].size(); i++)
		{
			string tmp = stripQuotes(jj["above"][i]["subject"]);
			ite = find(states.begin(),states.end(),tmp);
			if(ite == states.end())
			{
				//Add a new state
				states.push_back(tmp);
			}
			tmp = stripQuotes(jj["above"][i]["reference"]);
			ite = find(states.begin(),states.end(),tmp);
			if(ite == states.end())
			{
				//Add a new state
				states.push_back(tmp);
			}
		}

		//Gather neighbor pairs
		neig_to.resize(states.size(),vector<int>());

		//Gather 'to_left_of' & 'to_right_of' pairs
		to_left_of_.resize(states.size(),vector<int>());
		to_right_of_.resize(states.size(),vector<int>());

		for(unsigned int i = 0; i < jj["to_left_of"].size(); i++)
		{
			string sub = stripQuotes(jj["to_left_of"][i]["subject"]);
			string ref = stripQuotes(jj["to_left_of"][i]["reference"]);
			int sub_i = std::distance(states.begin(),find(states.begin(),states.end(),sub));
			int ref_i = std::distance(states.begin(),find(states.begin(),states.end(),ref));

			to_left_of_[ref_i].push_back(sub_i);
			to_right_of_[sub_i].push_back(ref_i);

			//Also add the pairs to the neighborhood relation list
			neig_to[sub_i].push_back(ref_i);
			neig_to[ref_i].push_back(sub_i);
		}

		//Gather 'above' & 'below' pairs
		above_.resize(states.size(),vector<int>());
		below_.resize(states.size(),vector<int>());

		for(unsigned int i = 0; i < jj["above"].size(); i++)
		{
			string sub = stripQuotes(jj["above"][i]["subject"]);
			string ref = stripQuotes(jj["above"][i]["reference"]);
			int sub_i = std::distance(states.begin(),find(states.begin(),states.end(),sub));
			int ref_i = std::distance(states.begin(),find(states.begin(),states.end(),ref));

			above_[ref_i].push_back(sub_i);
			below_[sub_i].push_back(ref_i);

			//Also add the pairs to the neighborhood relation list
			neig_to[sub_i].push_back(ref_i);
			neig_to[ref_i].push_back(sub_i);
		}
	}
	catch(std::exception &e)
	{
		cout << e.what() << endl;
		return false;
	}

	return true;
}

bool Neighborhood::save(string file_name)
{
	//To be implemented

	return true;
}

bool Neighborhood::load(string file_name)
{
	//To be implemented

	return true;
}

vector<string> Neighborhood::neigTo(string const &s, bool &success)
{
	vector<string>::iterator ite = find(states.begin(),states.end(),s);

	//Check if the state 's' actually exists
	if(ite == states.end())
	{
		success = false;
		return vector<string>();
	}
	else
	{
		//Get 's' index
		int index = std::distance(states.begin(), ite);

		//Gather all the neighbors of 's'
		vector<string> neig_vec;
		for(unsigned int i = 0; i < neig_to[index].size(); i++)
		{
			neig_vec.push_back(states[neig_to[index][i]]);
		}

		success = true;
		return neig_vec;
	}
}

vector<string> Neighborhood::neigToExc(string const &s, vector<string> const &exc)
{
	//Gather all of s' neighbors
	bool success;
	vector<string> tmp = neigTo(s,success);

	//If neighborhood was successfully gathered, filter out the exceptions
	if(success)
	{
		vector<string>::iterator ite;
		for(unsigned int i = 0; i < exc.size(); i++)
		{
			ite = find(tmp.begin(),tmp.end(),exc[i]);

			if(ite != tmp.end())
			{
				int index = std::distance(tmp.begin(),ite);
				tmp.erase(tmp.begin() + index);
			}
		}
	}

	return tmp;
}

vector<string> Neighborhood::leftOf(string const &s, bool &success)
{
	vector<string>::iterator ite = find(states.begin(),states.end(),s);

	//Check if the state 's' actually exists
	if(ite == states.end())
	{
		success = false;
		return vector<string>();
	}
	else
	{
		//Get 's' index
		int index = std::distance(states.begin(), ite);

		//Get all the states at the left of 's'
		vector<string> tmp;
		for(unsigned int i = 0; i < to_left_of_[index].size(); i++)
		{
			tmp.push_back(states[to_left_of_[index][i]]);
		}

		//Return the 'success' flag & vector of neighbors
		success = (tmp.size() > 0);
		return tmp;
	}
}

vector<string> Neighborhood::rightOf(string const &s, bool &success)
{
	vector<string>::iterator ite = find(states.begin(),states.end(),s);

	//Check if the state 's' actually exists
	if(ite == states.end())
	{
		success = false;
		return vector<string>();
	}
	else
	{
		//Get 's' index
		int index = std::distance(states.begin(), ite);

		//Get all the states at the right of 's'
		vector<string> tmp;
		for(unsigned int i = 0; i < to_right_of_[index].size(); i++)
		{
			tmp.push_back(states[to_right_of_[index][i]]);
		}

		//Return the 'success' flag & vector of neighbors
		success = (tmp.size() > 0);
		return tmp;
	}
}

vector<string> Neighborhood::aboveOf(string const &s, bool &success)
{
	vector<string>::iterator ite = find(states.begin(),states.end(),s);

	//Check if the state 's' actually exists
	if(ite == states.end())
	{
		success = false;
		return vector<string>();
	}
	else
	{
		//Get 's' index
		int index = std::distance(states.begin(), ite);

		//Get all the states above of 's'
		vector<string> tmp;
		for(unsigned int i = 0; i < above_[index].size(); i++)
		{
			tmp.push_back(states[above_[index][i]]);
		}

		//Return the 'success' flag & vector of neighbors
		success = (tmp.size() > 0);
		return tmp;
	}
}

vector<string> Neighborhood::belowOf(string const &s, bool &success)
{
	vector<string>::iterator ite = find(states.begin(),states.end(),s);

	//Check if the state 's' actually exists
	if(ite == states.end())
	{
		success = false;
		return vector<string>();
	}
	else
	{
		//Get 's' index
		int index = std::distance(states.begin(), ite);

		//Get all the states below of 's'
		vector<string> tmp;
		for(unsigned int i = 0; i < below_[index].size(); i++)
		{
			tmp.push_back(states[below_[index][i]]);
		}

		//Return the 'success' flag & vector of neighbors
		success = (tmp.size() > 0);
		return tmp;
	}
}

bool Neighborhood::propNeigNav(TreeHandle &th)
{
	//Check that the tree has at least 2 levels, besides root's level
	unsigned int depth = th.depth();
	if(depth <= 2) return false;

	//Save the amount of states stored previously to the propagation process
	unsigned pre_s_size = states.size();

	//Start the propagation of neighborhoods process
	vector<string> new_s,tmp;
	vector< vector<int> > new_l,new_a;
	for(unsigned int i = depth - 1; i > 1; i--)
	{
		//Clear the vectors that will store the relations gathered in the i-th level
		new_s.clear();
		new_l.clear();
		new_a.clear();

		//Get all the nodes at the i-th level
		tmp = th.keysAtLevel(i);

		for(unsigned int j = 0; j < tmp.size(); j++)
		{
			//Get the 'left' and 'above' neig. states
			bool l_res, a_res;
			vector<string> l_neig, a_neig;
			l_neig = leftOf(tmp[j],l_res);
			a_neig = aboveOf(tmp[j],a_res);

			//Get the j-th state's parent
			bool j_res;
			string j_par = th.parent(tmp[j],j_res);

			//Before adding a new relation pair, make sure that 
			//j-state's parent has already been added
			vector<string>::iterator ite;
			ite = find(new_s.begin(),new_s.end(),j_par);
			int j_par_index(-1);
			if(ite == new_s.end())
			{
				new_s.push_back(j_par);
				new_l.push_back(vector<int>());
				new_a.push_back(vector<int>());
				j_par_index = new_s.size() - 1;
			}
			else j_par_index = std::distance(new_s.begin(),ite);

			//Check among its left-neighbors if thay have different parents
			for(unsigned int k = 0; k < l_neig.size(); k++)
			{
				//Compare if they have different parents
				string l_par = th.parent(l_neig[k],j_res);

				//They have different parents
				if(j_res && l_par != j_par)
				{
					//First check if the l-parent has already been added in the vector of states
					ite = find(new_s.begin(),new_s.end(),l_par);
					int l_par_index(-1);
					if(ite == new_s.end())
					{
						//Add the l-parent node to the local vector of states
						new_s.push_back(l_par);
						new_l.push_back(vector<int>());
						new_a.push_back(vector<int>());
						l_par_index = new_s.size() - 1;
					}
					//Get l-parent node's index in the local vector of states
					else l_par_index = std::distance(new_s.begin(),ite);

					//Check if the 'to-left-of' pair has already been added
					vector<int>::iterator ite_i;
					ite_i = find(new_l[j_par_index].begin(),new_l[j_par_index].end(),l_par_index);
					if(ite_i == new_l[j_par_index].end())
					{
						//Add the 'to-left-of' relation between the parents
						new_l[j_par_index].push_back(l_par_index);
					}
				}
			}

			//Check among its above-neighbors if thay have different parents
			for(unsigned int k = 0; k < a_neig.size(); k++)
			{
				//Get the k-th above-neighbor's parent
				string a_par = th.parent(a_neig[k],j_res);

				//Compare if they have different parents
				//They have different parents
				if(j_res && a_par != j_par)
				{
					//First check if the a-parent has already been added in the vector of states
					ite = find(new_s.begin(),new_s.end(),a_par);
					int a_par_index(-1);
					if(ite == new_s.end())
					{
						//Add the a-parent node to the local vector of states
						new_s.push_back(a_par);
						new_l.push_back(vector<int>());
						new_a.push_back(vector<int>());
						a_par_index = new_s.size() - 1;
					}
					//Get a-parent node's index in the local vector of states
					else a_par_index = std::distance(new_s.begin(),ite);

					//Check if the 'above' pair has already been added
					vector<int>::iterator ite_i;
					ite_i = find(new_a[j_par_index].begin(),new_a[j_par_index].end(),a_par_index);
					if(ite_i == new_a[j_par_index].end())
					{
						//Add the 'above' relation between the parents
						new_a[j_par_index].push_back(a_par_index);
					}
				}
			}
		}

		//Shift index values before appending them to the global vectors
		int curr_states_size = states.size();
		for(unsigned int j = 0; j < new_l.size(); j++)
		{
			for(unsigned int k = 0; k < new_l[j].size(); k++) new_l[j][k] += curr_states_size;
			for(unsigned int k = 0; k < new_a[j].size(); k++) new_a[j][k] += curr_states_size;
		}

		//Append the the propagated neig-pairs and its states to the global vectors
		states.insert(states.end(),new_s.begin(),new_s.end());
		to_left_of_.insert(to_left_of_.end(),new_l.begin(),new_l.end());
		above_.insert(above_.end(),new_a.begin(),new_a.end());

		//Insert vectors of unassigned indexes to 'to-right-of' and 'below', so
		//their size match the one of 'states', 'to-left-of' and 'above'
		vector<vector<int> > non_index(new_s.size(),vector<int>());
		to_right_of_.insert(to_right_of_.end(),non_index.begin(),non_index.end());
		below_.insert(below_.end(),non_index.begin(),non_index.end());
	}

	//Generate the 'to-right-of' & 'below' pairs based on the 'to-left-of' & 'above' pairs
	for(unsigned int i = pre_s_size; i < states.size(); i++)
	{
		for(unsigned int j = 0; j < to_left_of_[i].size(); j++)
		{
			to_right_of_[to_left_of_[i][j]].push_back(i);
		}

		for(unsigned int j = 0; j < above_[i].size(); j++)
		{
			below_[above_[i][j]].push_back(i);
		}
	}

	//Update the neighbors vector
	for(unsigned int i = pre_s_size; i < states.size(); i++)
	{
		vector<int> tmp;
		for(unsigned int j = 0; j < to_left_of_[i].size(); j++) tmp.push_back(to_left_of_[i][j]);
		for(unsigned int j = 0; j < to_right_of_[i].size(); j++) tmp.push_back(to_right_of_[i][j]);
		for(unsigned int j = 0; j < above_[i].size(); j++) tmp.push_back(above_[i][j]);
		for(unsigned int j = 0; j < below_[i].size(); j++) tmp.push_back(below_[i][j]);

		neig_to.push_back(tmp);
	}

	return true;
}

bool Neighborhood::recPath(vector<string> &path, string const &target, int const &lbound)
{
	//The path must have at least one building as starter point
	if(path.size() == 0) return false;

	//Avoid exploring paths that are larger one  of the already found ones
	if(path.size() >= lbound && lbound != -1) return false;

	//Get the last building in the current path
	string b = path[path.size()-1];

	//Explore the paths that follow the neighbors of 'b'
	int c_lbound = lbound;
	bool r;
	vector<string> neig = neigTo(b,r);
	vector<vector<string> > paths;
	for(unsigned int i = 0; i < neig.size(); i++)
	{
		//Avoid cycles by not adding buildings that are already in the path
		vector<string>::iterator ite;
		ite = find(path.begin(),path.end(),neig[i]);
		if(ite != path.end()) continue;

		//Target has been reached
		if(neig[i] == target)
		{
			path.push_back(target);
			return true;
		}

		//Add neig[i] to keep exploring
		vector<string> c_path = path;
		c_path.push_back(neig[i]);
		if(recPath(c_path,target,c_lbound))
		{
			paths.push_back(c_path);
			if(c_lbound == -1 || c_path.size() < c_lbound) c_lbound = c_path.size();
		}
	}

	//If all the paths generated by following the neighbors are deadends
	//then, the current path is also a deadend
	if(paths.size() == 0) return false;
	else
	{
		//From the succesful paths, return the shortest one
		int p_idx(-1);
		int p_len(-1);
		for(unsigned int i = 0; i < paths.size(); i++)
		{
			if(p_len == -1 || paths[i].size() < p_len)
			{
				p_len = paths[i].size();
				p_idx = i;
			}
		}

		path = paths[p_idx];
		return true;
	}
}

bool Neighborhood::recPath(vector<string> const &subspace, vector<string> &path, string const &target, int const &lbound)
{
	//The path must have at least one building as starter point
	if(path.size() == 0) return false;

	//Avoid exploring paths that are larger one  of the already found ones
	if(path.size() >= lbound && lbound != -1) return false;

	//Get the last building in the current path
	string b = path[path.size()-1];

	//Explore the paths that follow the neighbors of 'b'
	int c_lbound = lbound;
	bool r;
	vector<string> neig = neigTo(b,r);
	vector<vector<string> > paths;
	for(unsigned int i = 0; i < neig.size(); i++)
	{
		//Avoid cycles by not adding buildings that are already in the path
		vector<string>::const_iterator ite;
		ite = find(path.begin(),path.end(),neig[i]);
		if(ite != path.end()) continue;

		//Avoid expanding with states that are not in the bounded sub-space
		ite = find(subspace.begin(),subspace.end(),neig[i]);
		if(ite == subspace.end()) continue;

		//Target has been reached
		if(neig[i] == target)
		{
			path.push_back(target);
			return true;
		}

		//Add neig[i] to keep exploring
		vector<string> c_path = path;
		c_path.push_back(neig[i]);
		if(recPath(c_path,target,c_lbound))
		{
			paths.push_back(c_path);
			if(c_lbound == -1 || c_path.size() < c_lbound) c_lbound = c_path.size();
		}
	}

	//If all the paths generated by following the neighbors are deadends
	//then, the current path is also a deadend
	if(paths.size() == 0) return false;
	else
	{
		//From the succesful paths, return the shortest one
		int p_idx(-1);
		int p_len(-1);
		for(unsigned int i = 0; i < paths.size(); i++)
		{
			if(p_len == -1 || paths[i].size() < p_len)
			{
				p_len = paths[i].size();
				p_idx = i;
			}
		}

		path = paths[p_idx];
		return true;
	}
}

