#include <TreeHandle.hpp>
#include <Neighborhood.hpp>

using namespace st_tree;
using json = nlohmann::json;
using namespace std;

string TreeHandle::stripQuotes(string const &s)
{
	string out_s("");
	for(int i = 0; i < s.length(); i++)
	{
		if(s[i] != '\"') out_s += s[i];
	}

	return out_s;
}

void TreeHandle::recSave(ofstream &outfile, unsigned d, tree<unsigned, keyed<string> >::node_type::iterator &ite)
{
	//Write the current node key with ':'s indicating its depth
	for(unsigned int i = 0; i < d; i++) outfile << ":";
	outfile << ite->key() << endl;

	//Iterate over the current node's children
	for(tree<unsigned, keyed<string> >::node_type::iterator ite_(ite->begin()); ite_ != ite->end(); ++ite_)
	{
		recSave(outfile,d+1,ite_);
	}
}

bool TreeHandle::recLoad(tree<unsigned, keyed<string> >::node_type &n, vector<string> const &vec, unsigned int &index)
{
	bool got_child(false);
	string last_child("");

	for(unsigned int i = index; i < vec.size(); i++)
	{
		int count(0);
		for(int j = 0; j < vec[i].length(); j++) if(vec[i][j] == ':') count++;

		//Insert a child to n
		if(count == (n.ply() + 1))
		{
			string sub = vec[i].substr(count);
			n.insert(sub,0);

			//Set flag & global var
			got_child = true;
			last_child = sub;
		}
		//Insert a child into one of n's last inserted child
		else if(count == (n.ply() + 2) && got_child)
		{
			//Insert granchildren of n through recursive calls
			bool sub_result = recLoad(n[last_child], vec, i);

			//Something has gone wrong & propagate the message to the first call
			if(!sub_result) return false;
		}
		//Insert a node that is not ascendent nor descendent of n
		else if(count <= n.ply() && count > 0)
		{
			index = i-1;
			return true;
		}
		//Unexpected structures
		else return false;
	}

	index = vec.size();
	return true;
}

TreeHandle::TreeHandle()
{
	t_.insert(0);
}

void TreeHandle::insert(string node_key, unsigned val)
{
	t_.root().insert(node_key,val);
}

bool TreeHandle::insert(string parent_key, string node_key, unsigned val)
{
	//First, find the parent node
	bool found_parent(false);
	for(tree<unsigned, keyed<string> >::iterator j(t_.begin()); j != t_.end(); ++j)
	{
		//Only if parent found, then insert the new node
		if(j->key() == parent_key)
		{
			j->insert(node_key,val);
			found_parent = true;
			break;
		}
	}

	return found_parent;
}

bool TreeHandle::query(string node_key, unsigned &output_val)
{
	//First find the requested node
	bool found_node(false);
	for(tree<unsigned, keyed<string> >::iterator j(t_.begin()); j != t_.end(); ++j)
	{
		//Save the node's data value
		if(j->key() == node_key)
		{
			output_val = j->data();
			found_node = true;
			break;
		}
	}

	return found_node;
}

bool TreeHandle::save(string file_name)
{
	//Create the output file
	ofstream outfile;
	outfile.open(file_name);
	if(!outfile.is_open())
	{
		cout << "ERROR[save]: could not open " + file_name << endl;
		return false;
	}

	//Write the elements from tree in the output file
	for(tree<unsigned, keyed<string> >::node_type::iterator i(t_.root().begin()); i != t_.root().end(); ++i)
	{
		recSave(outfile,1,i);
	}
	outfile.close();

	return true;
}

bool TreeHandle::load(string file_name)
{
	//Open the input file
	ifstream infile;
	infile.open(file_name);
	if(!infile.is_open())
	{
		cout << "ERROR[load]: could not open " + file_name << endl;
		return false;
	}

	//Extract the lines from the file
	string tmp_s;
	vector<string> vec;
	while(getline(infile,tmp_s)) vec.push_back(tmp_s);
	infile.close();

	//Create a local tree
	tree<unsigned, keyed<string>> t_local;
	t_local.insert(0);

	//Insert nodes from the file into the local tree
	unsigned int index(0);
	bool result = recLoad(t_local.root(), vec, index);

	//Only if the loading process succeeded, then save the gathered tree
	if(result) t_ = t_local;

	return result;
}

unsigned int TreeHandle::depth() const
{
	//Return the tree's max depth
	unsigned int d = t_.depth();

	return d;
}

vector<string> TreeHandle::keysAtLevel(unsigned int level)
{
	//Requested level is for root or out of range
	if(level == 0 || level >= t_.depth()) return vector<string>();

	//Gather those node's key that are at the requested level 
	vector<string> out_keys;
	for (tree<unsigned, keyed<string> >::iterator i(t_.begin());  i != t_.end();  ++i)
	{
		if(i->ply() == level)
		{
			out_keys.push_back(i->key());
		}
	}

	return out_keys;
}

vector<string> TreeHandle::keysOfChildren(string parent_key)
{
	vector<string> out_keys;

	//First, find the parent node
	for(tree<unsigned, keyed<string> >::iterator i(t_.begin()); i != t_.end(); ++i)
	{
		if(i->key() == parent_key)
		{
			//Gather all of its children
			for(tree<unsigned, keyed<string> >::node_type::iterator j(i->begin()); j != i->end(); ++j)
			{
				out_keys.push_back(j->key());
			}

			break;
		}
	}

	return out_keys;
}

bool TreeHandle::isParent(string parent_key, string child_key)
{
	bool is_parent(false);

	//First, find the child node
	for(tree<unsigned, keyed<string> >::iterator i(t_.begin()); i != t_.end(); ++i)
	{
		if(i->key() == child_key)
		{
			//Determine if its the parent node
			is_parent = (i->parent().key() == parent_key);
			break;
		}
	}

	return is_parent;
}

bool TreeHandle::isAncestor(string ancestor_key, string child_key)
{
	//First get the ancestor node
	bool is_ancestor(false);
	for(tree<unsigned, keyed<string> >::iterator i(t_.begin()); i != t_.end(); ++i)
	{
		if(i->key() == ancestor_key)
		{
			//Then find the child node
			for(tree<unsigned, keyed<string> >::iterator j(t_.begin()); j != t_.end(); ++j)
			{
				if(j->key() == child_key)
				{
					//Determine if it is the ancestor of child_node
					is_ancestor = i->is_ancestor(*j);

					break;
				}
			}

			break;
		}
	}

	return is_ancestor;
}

bool TreeHandle::inTree(string const &node_key)
{
	bool is_in(false);
	for (tree<unsigned, keyed<string> >::iterator i(t_.begin());  i != t_.end();  ++i)
	{
		if(i->key() == node_key)
		{
			is_in = true;
			break;
		}
	}

	return is_in;
}

int TreeHandle::ply(string const &node_key)
{
	int node_ply(-1);

	for (tree<unsigned, keyed<string> >::iterator i(t_.begin());  i != t_.end();  ++i)
	{
		if(i->key() == node_key)
		{
			node_ply = i->ply();
			break;
		}
	}

	return node_ply;
}

vector<string> TreeHandle::hieState(string const &node_key)
{
	vector<string> h_state;
	for (tree<unsigned, keyed<string> >::iterator i(t_.begin());  i != t_.end();  ++i)
	{
		if(i->key() == node_key)
		{
			vector<string> v_tmp;
			v_tmp.push_back(node_key);

			//Gather all of the node's ancestors
			tree<unsigned, keyed<string> >::node_type *node = &i->parent();
			while(true)
			{
				//When root is found, stop looking for ancestors
				if(node->is_root())
				{
					v_tmp.push_back("root");
					break;
				}
				else
				{
					v_tmp.push_back(node->key());
					node = &node->parent();
				}
			}

			//Invert the elements in the vector
			for(int j = v_tmp.size()-1; j > -1; j--) h_state.push_back(v_tmp[j]);

			break;
		}
	}

	return h_state;
}

tuple<string, int> TreeHandle::dca(string const &node_a, string const &node_b)
{
	//First make sure both nodes are in the tree
	bool found_a(false);
	bool found_b(false);
	tree<unsigned, keyed<string> >::iterator ite_a, ite_b;
	for(tree<unsigned, keyed<string> >::iterator i(t_.begin()); i != t_.end(); ++i)
	{
		if(i->key() == node_a)
		{
			ite_a = i;
			found_a = true;
		}

		if(i->key() == node_b)
		{
			ite_b = i;
			found_b = true;
		}

		if(found_a && found_b) break;
	}

	//Quit if at least one of them was not found
	if(!found_a || !found_b) return tuple<string, int>("",-1);

	//Compare their ancestors and get the common one with at the deepest level
	string dca_key("root");
	int dca_ply(0);
	tree<unsigned, keyed<string> >::node_type *tmp = &ite_a->parent();
	while(true)
	{
		//Root is the deepest common ancestor
		if(tmp->is_root()) break;

		//Found the deepest common ancestor that is not root
		if(tmp->is_ancestor(*ite_b))
		{
			dca_key = tmp->key();
			dca_ply = tmp->ply();
			break;
		}
		else tmp = &tmp->parent();
	}

	return tuple<string, int>(dca_key,dca_ply);
}

string TreeHandle::parent(string const &node_key, bool &success)
{
	success = false;
	string p_key("");
	for(tree<unsigned, keyed<string> >::iterator i(t_.begin());  i != t_.end();  ++i)
	{
		if(i->key() == node_key)
		{
			tree<unsigned, keyed<string> >::node_type *tmp = &i->parent();
			if(!tmp->is_root())
			{
				success = true;
				p_key = tmp->key();
			}

			break;
		}
	}

	return p_key;
}

vector<tuple<string,string,int> > TreeHandle::borPaths(Neighborhood &nh, string const &bor1, string const &bor2, string const &room)
{
	vector<tuple<string,string,int> > conns;

	//Check for a valid "room" and a valid pair of walls
	if(ply(room) != 2) return conns;
	if(bor1 == bor2) return conns;
	if(bor1!="l" && bor1!="r" && bor1!="u" && bor1!="d") return conns;
	if(bor2!="l" && bor2!="r" && bor2!="u" && bor2!="d") return conns;

	//Get all the cells in 'room'
	vector<string> cell;
	vector<string> ss = keysOfChildren(room);
	for(unsigned int i = 0; i < ss.size(); i++)
	{
		vector<string> tmp = keysOfChildren(ss[i]);
		cell.insert(cell.end(),tmp.begin(),tmp.end());
	}

	//Check if "room" is a hall
	vector<string>::iterator ite;
	ite = find(h_name.begin(),h_name.end(),room);
	if(ite != h_name.end())
	{
		//For halls l-r and r-l border pair are the only valid ones
		if(!(bor1 == "l" && bor2 == "r") && !(bor1 == "r" && bor2 == "l")) return conns;

		//Get the hall's dimensions in cells
		int hi = std::distance(h_name.begin(),ite);
		int h_width = h_dim[hi][0]*h_dim[hi][1]*h_dim[hi][2];
		int h_height = h_dim[hi][3]*h_dim[hi][4]*h_dim[hi][5];

		//Get the ID of the upper-left cell
		int ul(-1);
		for(unsigned int i = 0; i < cell.size(); i++)
		{
			int id = stoi(cell[i].substr(1));
			if(i == 0) ul = id;
			else if(id < ul) ul = id;
		}

		//Get the set of left & right boredring cells
		vector<string> l_bor,r_bor;
		for(int i = 0; i < h_height; i++)
		{
			l_bor.push_back("c" + to_string(ul + i*h_width));
			r_bor.push_back("c" + to_string(ul + (i+1)*h_width - 1));
		}

		//Compute the Manhattan distance between each pair of cell from opposite borders
		for(unsigned int i = 0; i < l_bor.size(); i++)
		{
			for(unsigned int j = 0; j < r_bor.size(); j++)
			{
				int diff = std::abs(static_cast<double>(i) - static_cast<double>(j));
				int dist = h_width + diff - 1;
				//l-r borders
				if(bor1 == "l") conns.push_back(tuple<string,string,int>(l_bor[i],r_bor[j],dist));
				//r-l borders
				else conns.push_back(tuple<string,string,int>(r_bor[i],l_bor[j],dist));
			}
		}

		return conns;
	}
	else
	{
		//"room" is not a hall

		//Get the upper-left cell id
		int ul(-1);
		for(unsigned int i = 0; i < cell.size(); i++)
		{
			int id = stoi(cell[i].substr(1));
			if(i == 0) ul = id;
			else if(id < ul) ul = id;
		}

		//Dimensions required to compute te coordinate of a cell
		//with respect to the upper-left cell
		bool res;
		string bb = parent(room,res);
		ite = find(b_name.begin(),b_name.end(),bb);
		int b_index = std::distance(b_name.begin(),ite);
		int b_width = b_dim[b_index][0]*b_dim[b_index][1]*b_dim[b_index][2];
		int r_width = b_dim[b_index][1]*b_dim[b_index][2];
		int r_height = b_dim[b_index][4]*b_dim[b_index][5];

		//Get the connection cells for each border wall
		vector<string> bc1, bc2;
		vector<tuple<int,int> > cc1, cc2;
		for(unsigned int i = 0; i < cell.size(); i++)
		{
			//Get the i-th cell's coordinates relative to the room
			int c_id = stoi(cell[i].substr(1));
			int x = (c_id - ul) % b_width;
			int y = (c_id - ul) / b_width;

			string border("");
			if(x == 0) border += "l";
			if(x == (r_width-1)) border += "r";
			if(y == 0) border += "u";
			if(y == (r_height-1)) border += "d";
			int i_1 = border.find(bor1);
			int i_2 = border.find(bor2);

			//Check if i-th cell connects with other room in border-1
			if(i_1 >= 0 && i_1 < border.length())
			{
				vector<string> tmp;
				if(bor1 == "l") tmp = nh.leftOf(cell[i],res);
				else if(bor1 == "r") tmp = nh.rightOf(cell[i],res);
				else if(bor1 == "u") tmp = nh.aboveOf(cell[i],res);
				else if(bor1 == "d") tmp = nh.belowOf(cell[i],res);

				//The i-th cell is bordering in border-1
				//Save the cell & its coordinates
				if(tmp.size() > 0)
				{
					bc1.push_back(cell[i]);
					cc1.push_back(tuple<int,int>(x,y));
				}
			}

			//Check if i-th cell connects with other room in border-2
			if(i_2 >= 0 && i_2 < border.length())
			{
				vector<string> tmp;
				if(bor2 == "l") tmp = nh.leftOf(cell[i],res);
				else if(bor2 == "r") tmp = nh.rightOf(cell[i],res);
				else if(bor2 == "u") tmp = nh.aboveOf(cell[i],res);
				else if(bor2 == "d") tmp = nh.belowOf(cell[i],res);

				//The i-th cell is bordering in border-2
				//Save the cell & its coordinates
				if(tmp.size() > 0)
				{
					bc2.push_back(cell[i]);
					cc2.push_back(tuple<int,int>(x,y));
				}
			}
		}

		//If one of the requested borders does not connect with
		//other room return an empty vector
		if(bc1.size() == 0 || bc2.size() == 0) return conns;

		//Compute the distance for pair of cells from
		//different borders
		for(unsigned int i = 0; i < cc1.size(); i++)
		{
			for(unsigned int j = 0; j < cc2.size(); j++)
			{
				int dist = abs(get<0>(cc1[i]) - get<0>(cc2[j])) + abs(get<1>(cc1[i]) - get<1>(cc2[j]));
				tuple<string,string,int> tmp(bc1[i],bc2[j],dist);
				conns.push_back(tmp);
			}
		}

		return conns;
	}
}

vector<tuple<string,string,int> > TreeHandle::cellBorPaths(Neighborhood &nh, string const &init_cell, string const &bor, string const &room)
{
	vector<tuple<string,string,int> > conns;

	//Check for a valid "room" and a valid wall-side
	if(ply(room) != 2) return conns;
	if(bor!="l" && bor!="r" && bor!="u" && bor!="d") return conns;
	if(!isAncestor(room,init_cell)) return conns;

	//Get all the cells in 'room'
	vector<string> cell;
	vector<string> ss = keysOfChildren(room);
	for(unsigned int i = 0; i < ss.size(); i++)
	{
		vector<string> tmp = keysOfChildren(ss[i]);
		cell.insert(cell.end(),tmp.begin(),tmp.end());
	}

	//Check if "room" is a hall
	vector<string>::iterator ite;
	ite = find(h_name.begin(),h_name.end(),room);
	if(ite != h_name.end())
	{
		//For halls l & r are the only valid borders
		if(bor == "l" && bor == "r") return conns;

		//Get the hall's dimensions in cells
		int hi = std::distance(h_name.begin(),ite);
		int h_width = h_dim[hi][0]*h_dim[hi][1]*h_dim[hi][2];
		int h_height = h_dim[hi][3]*h_dim[hi][4]*h_dim[hi][5];

		//Get the ID of the upper-left cell
		int ul(-1);
		for(unsigned int i = 0; i < cell.size(); i++)
		{
			int id = stoi(cell[i].substr(1));
			if(i == 0) ul = id;
			else if(id < ul) ul = id;
		}

		//Get the set of boredring cells
		vector<string> x_bor;
		for(int i = 0; i < h_height; i++)
		{
			if(bor == "l") x_bor.push_back("c" + to_string(ul + i*h_width));
			else x_bor.push_back("c" + to_string(ul + (i+1)*h_width - 1));
		}

		//Compute the Manhattan distance between the cell & every border cell 
		int c_idx = stoi(init_cell.substr(1));
		int cx = (c_idx - ul) % h_width;
		int cy = (c_idx - ul) / h_width;
		for(unsigned int i = 0; i < x_bor.size(); i++)
		{
			int b_idx = stoi(x_bor[i].substr(1));
			int bx = (b_idx - ul) % h_width;
			int by = (b_idx - ul) / h_width;
			int dist = abs(cx - bx) + abs(cy - by);
			conns.push_back(tuple<string,string,int>(init_cell,x_bor[i],dist));
		}

		return conns;
	}
	else
	{
		//"room" is not a hall

		//Get the upper-left cell id
		int ul(-1);
		for(unsigned int i = 0; i < cell.size(); i++)
		{
			int id = stoi(cell[i].substr(1));
			if(i == 0) ul = id;
			else if(id < ul) ul = id;
		}

		//Dimensions required to compute te coordinate of a cell
		//with respect to the upper-left cell
		bool res;
		string bb = parent(room,res);
		ite = find(b_name.begin(),b_name.end(),bb);
		int b_index = std::distance(b_name.begin(),ite);
		int b_width = b_dim[b_index][0]*b_dim[b_index][1]*b_dim[b_index][2];
		int r_width = b_dim[b_index][1]*b_dim[b_index][2];
		int r_height = b_dim[b_index][4]*b_dim[b_index][5];

		//Get the connection cells for each border wall
		vector<string> bc;
		vector<tuple<int,int> > cc;
		for(unsigned int i = 0; i < cell.size(); i++)
		{
			//Get the i-th cell's coordinates relative to the room
			int c_id = stoi(cell[i].substr(1));
			int x = (c_id - ul) % b_width;
			int y = (c_id - ul) / b_width;

			string border("");
			if(x == 0) border += "l";
			if(x == (r_width-1)) border += "r";
			if(y == 0) border += "u";
			if(y == (r_height-1)) border += "d";
			int i_1 = border.find(bor);

			//Check if i-th cell connects with other room in border
			if(i_1 >= 0 && i_1 < border.length())
			{
				vector<string> tmp;
				if(bor == "l") tmp = nh.leftOf(cell[i],res);
				else if(bor == "r") tmp = nh.rightOf(cell[i],res);
				else if(bor == "u") tmp = nh.aboveOf(cell[i],res);
				else if(bor == "d") tmp = nh.belowOf(cell[i],res);

				//The i-th cell is bordering in border-1
				//Save the cell & its coordinates
				if(tmp.size() > 0)
				{
					bc.push_back(cell[i]);
					cc.push_back(tuple<int,int>(x,y));
				}
			}
		}

		//If the requested border does not connect with
		//other room then return an empty vector
		if(bc.size() == 0) return conns;

		//Get the requested cell's coordinates relative to the room
		int c_id = stoi(init_cell.substr(1));
		int x = (c_id - ul) % b_width;
		int y = (c_id - ul) / b_width;

		//Compute the distance the requested cell and
		//every border cell
		for(unsigned int i = 0; i < cc.size(); i++)
		{
			int dist = abs(get<0>(cc[i]) - x) + abs(get<1>(cc[i]) - y);
			tuple<string,string,int> tmp(init_cell,bc[i],dist);
			conns.push_back(tmp);
		}

		return conns;
	}
}

int TreeHandle::manhDist(Neighborhood &nh, string const &sa,string const &sb)
{
	vector<string> hsa = hieState(sa);
	vector<string> hsb = hieState(sb);
	//'sa' & 'sb' must be both cells
	if(hsa.size() != 5 || hsb.size() != 5) return -1;
	//'sa' & 'sb' must be in the same room
	if(hsa[2] != hsb[2]) return -1;

	//Get all cells in the room
	vector<string> cell;
	vector<string> sub = keysOfChildren(hsa[2]);
	for(unsigned int i = 0; i < sub.size(); i++)
	{
		vector<string> tmp = keysOfChildren(sub[i]);
		cell.insert(cell.end(),tmp.begin(),tmp.end());
	}

	//Get the ID of the upper-left cell in the room
	int ul(-1);
	for(unsigned int i = 0; i < cell.size(); i++)
	{
		int id = stoi(cell[i].substr(1));
		if(i == 0 || id < ul) ul = id;
	}

	//Check if their room is a hall or not
	vector<string>::iterator ite;
	ite = find(h_name.begin(),h_name.end(),hsa[2]);
	int r_width(-1);
	//hall room
	if(ite != h_name.end())
	{
		//Get the width of the hall
		int h_idx = distance(h_name.begin(),ite);
		r_width = h_dim[h_idx][0]*h_dim[h_idx][1]*h_dim[h_idx][2];
	}
	//not-hall room
	else
	{
		ite = find(b_name.begin(),b_name.end(),hsa[1]);
		int b_idx = distance(b_name.begin(),ite);
		r_width = b_dim[b_idx][0]*b_dim[b_idx][1]*b_dim[b_idx][2];
	}

	//Get the coordinates of both cells
	int ax = (stoi(sa.substr(1)) - ul) % r_width;
	int ay = (stoi(sa.substr(1)) - ul) / r_width;
	int bx = (stoi(sb.substr(1)) - ul) % r_width;
	int by = (stoi(sb.substr(1)) - ul) / r_width;

	//Compute the Manhattan distance
	int dist = abs(ax - bx) + abs(ay -by);

	return dist;
}

bool TreeHandle::recRoomPath(Neighborhood &nh, vector<tuple<string,string,string,int> > &path, vector<string> const &tgt_hc, int const &ubound)
{
	//The path must have at least one room as starter point
	if(path.size() == 0) return false;

	//Make sure the upper-bound is not surpassed
	if(ubound != -1)
	{
		int sum(0);
		for(unsigned int i = 0; i < path.size(); i++)
		{
			if(get<3>(path[i]) != -1) sum += get<3>(path[i]);
		}

		if(sum > ubound) return false;
	}

	//Get the last room in the current path
	string lr = get<0>(path[path.size()-1]);
	string lc = get<1>(path[path.size()-1]);

	//Check if the room 'lr' contains the target cell
	if(lr == tgt_hc[2])
	{
		int dist = manhDist(nh,lc,tgt_hc[4]);
		get<2>(path[path.size()-1]) = tgt_hc[4];
		get<3>(path[path.size()-1]) = dist;
		return true;
	}

	//Get the border cells
	vector<string> tag,alltag;
	vector<tuple<string,string,int> > lb,rb,ub,db,allb;
	lb = cellBorPaths(nh,lc,"l",lr);
	rb = cellBorPaths(nh,lc,"r",lr);
	ub = cellBorPaths(nh,lc,"u",lr);
	db = cellBorPaths(nh,lc,"d",lr);

	//Put them all in a single vector
	tag = vector<string>(lb.size(),"l");
	alltag.insert(alltag.end(),tag.begin(),tag.end());
	allb.insert(allb.end(),lb.begin(),lb.end());
	tag = vector<string>(rb.size(),"r");
	alltag.insert(alltag.end(),tag.begin(),tag.end());
	allb.insert(allb.end(),rb.begin(),rb.end());
	tag = vector<string>(ub.size(),"u");
	alltag.insert(alltag.end(),tag.begin(),tag.end());
	allb.insert(allb.end(),ub.begin(),ub.end());
	tag = vector<string>(db.size(),"d");
	alltag.insert(alltag.end(),tag.begin(),tag.end());
	allb.insert(allb.end(),db.begin(),db.end());

	//Get the room & cell neighbors from each direction
	vector<string>::iterator ite;
	vector<string> vis;
	vector<tuple<string,string,int> > neig_r;
	for(unsigned int i = 0; i < allb.size(); i++)
	{
		bool res;
		vector<string> tmp;
		if(alltag[i] == "l") tmp = nh.leftOf(get<1>(allb[i]),res);
		if(alltag[i] == "r") tmp = nh.rightOf(get<1>(allb[i]),res);
		if(alltag[i] == "u") tmp = nh.aboveOf(get<1>(allb[i]),res);
		if(alltag[i] == "d") tmp = nh.belowOf(get<1>(allb[i]),res);
		tmp = hieState(tmp[0]);

		//Avoid exploring rooms that are already part of the current path
		//or that belong to a different building
		if(tmp[1] != tgt_hc[1]) continue;
		bool skip(false);
		for(unsigned int j = 0; j < path.size(); j++)
		{
			if(tmp[2] == get<0>(path[j]))
			{
				skip = true;
				break;
			}
		}
		if(skip) continue;

		//neig-room, neig-cell, index-to-allb
		neig_r.push_back(tuple<string,string,int>(tmp[2],tmp[4],i));
		ite = find(vis.begin(),vis.end(),tmp[2]);
		if(ite == vis.end()) vis.push_back(tmp[2]);
	}

	//Get the shortest path to traverse to each neighbor room
	vector<int> short2room;
	for(unsigned int i = 0; i < vis.size(); i++)
	{
		int min_d(-1);
		int min_i(-1);
		for(unsigned int j = 0; j < neig_r.size(); j++)
		{
			if(vis[i] == get<0>(neig_r[j]))
			{
				int d2border = get<2>(allb[get<2>(neig_r[j])]);
				if(min_d == -1 || d2border < min_d)
				{
					min_d = d2border;
					min_i = j;
				}
			}
		}

		short2room.push_back(min_i);
	}

	//Explore the rooms in 'vis'
	int loc_bound = ubound;
	vector<vector<tuple<string,string,string,int> > > paths;
	vector<int> p_sum;
	for(unsigned int i = 0; i < neig_r.size(); i++)
	{
		vector<tuple<string,string,string,int> > tmp;
		tmp = path;
		get<2>(tmp[tmp.size()-1]) = get<1>(allb[get<2>(neig_r[i])]);
		get<3>(tmp[tmp.size()-1]) = get<2>(allb[get<2>(neig_r[i])]);
		tmp.push_back(tuple<string,string,string,int>(get<0>(neig_r[i]),get<1>(neig_r[i]),"",-1));

		if(recRoomPath(nh,tmp,tgt_hc,loc_bound))
		{

			int sum(0);
			for(unsigned int j = 0; j < tmp.size(); j++) sum += get<3>(tmp[j]);

			//Only keep the shortest path so found far
			if(loc_bound == -1 || sum < loc_bound)
			{
				paths.push_back(tmp);
				p_sum.push_back(sum);
				loc_bound = sum;
			}
		}
	}

	if(paths.size() > 0)
	{
		//Return the shortest path found
		int min_s(-1);
		int min_i(-1);
		for(unsigned int i = 0; i < p_sum.size(); i++)
		{
			if(min_s == -1 || p_sum[i] < min_s)
			{
				min_s = p_sum[i];
				min_i = i;
			}
		}

		path = paths[min_i];
		return true;
	}
	else return false;
}

vector<vector<tuple<string,string,int,string> > > TreeHandle::roomPath(Neighborhood &nh,string const &init_cell,string const &target_room,string const &target_sc)
{
	vector<vector<tuple<string,string,int,string> > > final_paths;
	bool is_side(false);

	//Initial checking
	vector<string>::iterator ite;
	vector<string> hic = hieState(init_cell);
	vector<string> htr = hieState(target_room);
	if(hic.size() != 5) return final_paths;
	if(htr.size() != 3) return final_paths;
	if(hic[1] != htr[1]) return final_paths;
	if(target_sc != "l" && target_sc != "r" && target_sc != "u" && target_sc != "d")
	{
		vector<string> tmp = hieState(target_sc);
		if(tmp.size() != 5) return final_paths;
		if(tmp[2] != htr[2]) return final_paths;
	}
	else is_side = true;

	vector<string> tgt_c;
	if(is_side)
	{
		//Get every border cell of teh requested side
		vector<string> tmp = keysOfChildren(target_room);
		tmp = keysOfChildren(tmp[0]);
		vector<tuple<string,string,int> > tmp2 = cellBorPaths(nh,tmp[0],target_sc,target_room);
		for(unsigned int i = 0; i < tmp2.size(); i++)
		{
			tgt_c.push_back(get<1>(tmp2[i]));
		}
	}
	else tgt_c.push_back(target_sc);

	//Compute the shortest cell-wise path from 'init_cell' to each cell in 'tgt_c'
	for(unsigned int i = 0; i < tgt_c.size(); i++)
	{
		vector<string> htc = hieState(tgt_c[i]);
		vector<tuple<string,string,string,int> > tmp;
		tmp.push_back(tuple<string,string,string,int>(hic[2],hic[4],"",-1));
		recRoomPath(nh,tmp,htc);

		vector<tuple<string,string,int,string> > vec;
		for(unsigned int j = 0; j < tmp.size(); j++)
		{
			string side_cell("");

			//Get the neighbor-cell in the next building, for 'is_side' cells
			if(is_side && j == (tmp.size()-1))
			{
				bool res;
				vector<string> bor;
				if(target_sc == "l") bor = nh.leftOf(get<2>(tmp[j]),res);
				if(target_sc == "r") bor = nh.rightOf(get<2>(tmp[j]),res);
				if(target_sc == "u") bor = nh.aboveOf(get<2>(tmp[j]),res);
				if(target_sc == "d") bor = nh.belowOf(get<2>(tmp[j]),res);
				side_cell = bor[0];
			}

			tuple<string,string,int,string> tt(get<1>(tmp[j]),get<2>(tmp[j]),get<3>(tmp[j]),side_cell);
			vec.push_back(tt);
		}
		final_paths.push_back(vec);
	}

	return final_paths;
}

int TreeHandle::optPath(Neighborhood &nh, string const &sa, string const &sb, bool const &display)
{
	//Check for a valid pair of states
	int dlvl(-1);
	vector<string> hsa = hieState(sa);
	vector<string> hsb = hieState(sb);
	if(hsa.size() != 5 || hsb.size() != 5) return -1;
	for(unsigned int i = 0; i < hsa.size(); i++)
	{
		if(hsa[i] != hsb[i])
		{
			dlvl = i;
			break;
		}
	}

	//sa & sb are the same cell
	if(dlvl == -1) return 0; 

	//Building-wise path
	vector<string> b_path;

	//sa & sb are in different buildings
	if(dlvl == 1)
	{
		//Compute the shortest building-wise path
		b_path.push_back(hsa[1]);
		nh.recPath(b_path,hsb[1]);
	}
	//sa & sb are in the same building
	else b_path.push_back(hsa[1]);

	//all_paths: path in building, cell-pairs
	vector<vector<tuple<string,string,int,string> > > all_paths;
	vector<int> paths_dist;
	vector<string> curr_init;
	vector<string>::iterator ite;
	for(unsigned int i = 0; i < (b_path.size()-1); i++)
	{
		//Get the initial cells for 
		curr_init.clear();
		if(i == 0) curr_init.push_back(sa);
		else
		{
			int l_b = all_paths.size() - 1;
			for(unsigned int j = 0; j < all_paths[l_b].size(); j++)
			{
				int l_p = all_paths[l_b].size() - 1;
				string tmp = get<3>(all_paths[l_b][l_p]);
				curr_init.push_back(tmp);
			}
		}

		//Get the i-th to (i+1)-th building direction
		bool res;
		vector<string> tmp;
		string b_dir("");
		tmp = nh.leftOf(b_path[i],res);
		ite = find(tmp.begin(),tmp.end(),b_path[i+1]);
		if(ite != tmp.end()) b_dir = "l";
		tmp = nh.rightOf(b_path[i],res);
		ite = find(tmp.begin(),tmp.end(),b_path[i+1]);
		if(ite != tmp.end()) b_dir = "r";
		tmp = nh.aboveOf(b_path[i],res);
		ite = find(tmp.begin(),tmp.end(),b_path[i+1]);
		if(ite != tmp.end()) b_dir = "u";
		tmp = nh.belowOf(b_path[i],res);
		ite = find(tmp.begin(),tmp.end(),b_path[i+1]);
		if(ite != tmp.end()) b_dir = "d";

		//Get the the list of rooms of the i-th building that connect with the next room
		vector<string> rooms = keysOfChildren(b_path[i]);
		vector<string> next_rooms = keysOfChildren(b_path[i+1]);
		vector<int> border_r;
		for(unsigned int j = 0; j < rooms.size(); j++)
		{
			if(b_dir == "l") tmp = nh.leftOf(rooms[j],res);
			if(b_dir == "r") tmp = nh.rightOf(rooms[j],res);
			if(b_dir == "u") tmp = nh.aboveOf(rooms[j],res);
			if(b_dir == "d") tmp = nh.belowOf(rooms[j],res);

			for(unsigned int k = 0; k < tmp.size(); k++)
			{
				ite = find(next_rooms.begin(),next_rooms.end(),tmp[k]);
				if(ite != next_rooms.end())
				{
					border_r.push_back(j);
					break;
				}
			}
		}

		//Compute the shortest path between every pair of init and
		//bordering to next-building cell
		vector<vector<tuple<string,string,int,string> > > tru_paths;
		for(unsigned int j = 0; j < curr_init.size(); j++)
		{
			for(unsigned int k = 0; k < border_r.size(); k++)
			{
				auto tmp2 = roomPath(nh,curr_init[j],rooms[border_r[k]],b_dir);
				tru_paths.insert(tru_paths.end(),tmp2.begin(),tmp2.end());
			}
		}

		//From the paths in 'tru_paths' select the shortest one
		int min_i(-1);
		int min_d(-1);
		for(unsigned int j = 0; j < tru_paths.size(); j++)
		{
			//Compute the j-th path's length
			int dist(0);
			for(unsigned int k = 0; k < tru_paths[j].size(); k++)
			{
				dist += get<2>(tru_paths[j][k]);
				if(k != (tru_paths[j].size() - 1)) dist++;
			}

			if(j == 0 || dist < min_d)
			{
				min_i = j;
				min_d = dist;
			}
		}

		//Save the shortest path
		all_paths.push_back(tru_paths[min_i]);
		paths_dist.push_back(min_d);
	}

	//Compute the shortest path to 'sb' in the last building
	string init_c;
	if(b_path.size() == 1) init_c = sa;
	else
	{
		int lpi = all_paths.size() - 1;
		int lci = all_paths[lpi].size() - 1;
		init_c = get<3>(all_paths[lpi][lci]);
	}
	vector<vector<tuple<string,string,int,string> > > tmp2;
	tmp2 = roomPath(nh,init_c,hsb[2],sb);
	int dist(0);
	for(unsigned int i = 0; i < tmp2[0].size(); i++)
	{
		dist += get<2>(tmp2[0][i]);
		if(i != (tmp2[0].size() - 1)) dist++;
	}
	all_paths.push_back(tmp2[0]);
	paths_dist.push_back(dist);

	//For display purposes
	if(display)
	{
		for(unsigned int i = 0; i < all_paths.size(); i++)
		{
			cout << "Traversing building " + b_path[i] << endl;
			for(unsigned int j = 0; j < all_paths[i].size(); j++)
			{
				cout << "\t"+get<0>(all_paths[i][j])+" -> "+get<1>(all_paths[i][j])+": ";
				cout << get<2>(all_paths[i][j]) << endl;

				if(j < (all_paths[i].size()-1))
				{
					cout << "\t"+get<1>(all_paths[i][j])+" -> "+get<0>(all_paths[i][j+1])+": ";
					cout << 1 << endl;
				}

				if(j == (all_paths[i].size()-1) && get<3>(all_paths[i][j]) != "")
				{
					cout << "\t"+get<1>(all_paths[i][j])+" -> "+get<3>(all_paths[i][j])+": ";
					cout << 1 << endl;
				}
			}
		}
	}

	//Compute the whole path's length
	int total_d(0);
	for(unsigned int i = 0; i < paths_dist.size(); i++)
	{
		total_d += paths_dist[i];
		if(i != (paths_dist.size()-1)) total_d++;
	}

	return total_d;
}

bool TreeHandle::navFromJson(string file_json)
{
	//Check for a valid file
	ifstream infile(file_json);
	if(!infile.is_open())
	{
		cout << "ERROR[navFromJson]: could not open " + file_json << endl;
		return false;
	}

	//Parse the file into a JSON structure
	json jj;
	infile >> jj;
	infile.close();

	//Create a local tree
	tree<unsigned, keyed<string>> t_local;
	t_local.insert(0);

	//try-catch block in case the json structure does not have the expected fields
	try
	{
		//Parse as a tree structure the JSON file
		for(unsigned int i = 0; i < jj["Buildings"].size(); i++)
		{
			//Insert the i-th building
			string tmp = stripQuotes(jj["Buildings"][i]["Name"]);
			t_local.root().insert(tmp, 0);

			//Save the building's dimensions
			b_name.push_back(tmp);
			vector<unsigned> tmp_dim = jj["Buildings"][i]["Dims"];
			b_dim.push_back(tmp_dim);

			//Get a pointer to the 'building' just added
			tree<unsigned, keyed<string> >::node_type::iterator i_ite;
			for(tree<unsigned, keyed<string> >::node_type::iterator ite(t_local.root().begin());
			    ite != t_local.root().end(); ++ite)
			{
				if(ite->key() == tmp)
				{
					i_ite = ite;
					break;
				}
			}

			//auto i_node = t_local.root()[tmp];
			auto i_vec = jj["Buildings"][i]["Rooms"];
			for(unsigned int j = 0; j < i_vec.size(); j++)
			{
				//Insert the j-th room
				string tmp = stripQuotes(i_vec[j]["Name"]);
				i_ite->insert(tmp,0);

				//Save the j-th room's dimensions if it is a hall to save its dimensions
				if(i_vec[j]["is_hall"])
				{
					h_name.push_back(tmp);
					vector<unsigned> tmp_dim2 = i_vec[j]["Dims"];
					h_dim.push_back(tmp_dim2);
				}

				//Get a pointer to the 'room' just added
				tree<unsigned, keyed<string> >::node_type::iterator j_ite;
				for(tree<unsigned, keyed<string> >::node_type::iterator ite(i_ite->begin());
				    ite != i_ite->end(); ++ite)
				{
					if(ite->key() == tmp)
					{
						j_ite = ite;
						break;
					}
				}

				//auto j_node = i_node[tmp];
				auto j_vec = i_vec[j]["Subsections"];
				for(unsigned int k = 0; k < j_vec.size(); k++)
				{
					//Insert the k-th subsection
					string tmp = stripQuotes(j_vec[k]["Name"]);
					j_ite->insert(tmp,0);

					//Get a pointer to the 'subsection' just added
					tree<unsigned, keyed<string> >::node_type::iterator k_ite;
					for(tree<unsigned, keyed<string> >::node_type::iterator ite(j_ite->begin());
					    ite != j_ite->end(); ++ite)
					{
						if(ite->key() == tmp)
						{
							k_ite = ite;
							break;
						}
					}

					//auto k_node = j_node[tmp];
					auto k_vec = j_vec[k]["Cells"];
					for(unsigned int l = 0; l < k_vec.size(); l++)
					{
						//Insert the l-th cell
						string tmp = stripQuotes(k_vec[l]["Name"]);
						k_ite->insert(tmp,0);
					}
				}
			}
		}
	}
	catch(st_tree::exception &e)
	{
		cout << e.what() << endl;
		return false;
	}

	//The parsing has succeeded
	t_ = t_local;
	return true;
}

