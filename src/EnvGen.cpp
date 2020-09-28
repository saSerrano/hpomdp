#include <EnvGen.hpp>

using namespace std;
using namespace cv;
using json = nlohmann::json;

bool EnvGen::loadConfig(string const &config_file)
{
	ifstream infile(config_file);
	if(!infile.is_open()) return false;

	//Read the configuration parameters from a JSON file
	json jj;
	try
	{
		//Parse the file into a JSON
		infile >> jj;
		infile.close();

		//Extract the amount of buildings and use-hall flag
		unsigned int n_b = static_cast<unsigned int>(jj["n_b"]);
		unsigned int hall_flag = static_cast<unsigned int>(jj["hall_flag"]);

		//Extract the use-hall probability & room-connection ratio
		float u_h_p = static_cast<float>(jj["use_hall_prob"]);
		float r_c_r = static_cast<float>(jj["room_conn_ratio"]);

		//Extract the width & height (in pixels) to be used when cells are drawn
		int cell_pixel_dim = static_cast<int>(jj["cell_pixel_dim"]);

		//Extract the vectors of dimension ranges
		vector<unsigned int> b_dims;
		vector<unsigned int> h_dims;
		vector<unsigned int> r_dims;
		vector<unsigned int> s_dims;
		for(unsigned int i = 0; i < jj["b_dims"].size(); i++)
		{
			b_dims.push_back(static_cast<unsigned int>(jj["b_dims"][i]));
		}
		for(unsigned int i = 0; i < jj["h_dims"].size(); i++)
		{
			h_dims.push_back(static_cast<unsigned int>(jj["h_dims"][i]));
		}
		for(unsigned int i = 0; i < jj["r_dims"].size(); i++)
		{
			r_dims.push_back(static_cast<unsigned int>(jj["r_dims"][i]));
		}
		for(unsigned int i = 0; i < jj["s_dims"].size(); i++)
		{
			s_dims.push_back(static_cast<unsigned int>(jj["s_dims"][i]));
		}

		//Evaluate the extracted parameters
		if(hall_flag > 2) return false;
		if(u_h_p < 0 || u_h_p > 1) return false;
		if(r_c_r < 0 || r_c_r > 1) return false;
		if(cell_pixel_dim <= 0) return false;
		if(b_dims.size() != 4) return false;
		if(h_dims.size() != 4) return false;
		if(r_dims.size() != 4) return false;
		if(s_dims.size() != 4) return false;
		if(b_dims[0] > b_dims[1] || b_dims[2] > b_dims[3]) return false;
		if(h_dims[0] > h_dims[1] || h_dims[2] > h_dims[3]) return false;
		if(r_dims[0] > r_dims[1] || r_dims[2] > r_dims[3]) return false;
		if(s_dims[0] > s_dims[1] || s_dims[2] > s_dims[3]) return false;

		//Save the extracted parameters
		this->n_b = n_b;
		this->hall_flag = hall_flag;
		this->USE_HALL_PROB = u_h_p;
		this->ROOM_CONN_RATIO = r_c_r;
		this->cell_pixel_dim = cell_pixel_dim;
		this->b_dims = b_dims;
		this->h_dims = h_dims;
		this->r_dims = r_dims;
		this->s_dims = s_dims;
	}
	catch(std::exception &e)
	{
		cout << e.what() << endl;
		return false;
	}

	return true;
}

unsigned int EnvGen::randomPick(unsigned int const &l_bound, unsigned int const &u_bound)
{
	if(l_bound == u_bound) return l_bound;

	int len = static_cast<int>(u_bound) - static_cast<int>(l_bound) + 1;
	int r = rand() % len + static_cast<int>(l_bound);

	return static_cast<unsigned int>(r);
}

int EnvGen::bHeight(building const &b)
{
	auto tmp = get<3>(b);
	int n_cells = tmp.size();
	vector<unsigned> dims = get<4>(b);

	return n_cells / (dims[0]*dims[1]*dims[2]);
}

EnvGen::EnvGen()
{
	n_b = 1;
	hall_flag = DONT_USE_HALLS;
	b_dims = vector<unsigned>(4,2);
	h_dims = vector<unsigned>(4,2);
	r_dims = vector<unsigned>(4,2);
	s_dims = vector<unsigned>(4,2);
	USE_HALL_PROB = 0.5;
	ROOM_CONN_RATIO = 0.5;
	cell_pixel_dim = 30;

	//Initialize random seed
	srand(time(NULL));
}

EnvGen::EnvGen(string const &config_file)
{
	bool load_success = loadConfig(config_file);

	//Some or both of the input parameters are not acceptable, then use default values
	if(!load_success)
	{
		n_b = 1;
		hall_flag = DONT_USE_HALLS;
		b_dims = vector<unsigned>(4,2);
		h_dims = vector<unsigned>(4,2);
		r_dims = vector<unsigned>(4,2);
		s_dims = vector<unsigned>(4,2);
		USE_HALL_PROB = 0.5;
		ROOM_CONN_RATIO = 0.5;
		cell_pixel_dim = 30;
	}

	//Initialize random seed
	srand(time(NULL));
}

unsigned int EnvGen::getNB() const
{
	return n_b;
}

unsigned int EnvGen::getHF() const
{
	return hall_flag;
}

vector<unsigned int> EnvGen::getBDim() const
{
	return b_dims;
}

vector<unsigned int> EnvGen::getHDim() const
{
	return h_dims;
}

vector<unsigned int> EnvGen::getRDim() const
{
	return r_dims;
}

vector<unsigned int> EnvGen::getSDim() const
{
	return s_dims;
}

void EnvGen::setNB(unsigned int n_b)
{
	this->n_b = n_b;
}

bool EnvGen::setHF(unsigned int hall_flag)
{
	if(hall_flag <= 2)
	{
		this->hall_flag = hall_flag;
		return true;
	}
	else return false;
}

bool EnvGen::setBDim(vector<unsigned int> const &dim)
{
	if(dim.size() != 4) return false;
	if(dim[0] > dim[1] || dim[2] > dim[3]) return false;

	b_dims = dim;
	return true;
}

bool EnvGen::setHDim(vector<unsigned int> const &dim)
{
	if(dim.size() != 4) return false;
	if(dim[0] > dim[1] || dim[2] > dim[3]) return false;

	h_dims = dim;
	return true;
}

bool EnvGen::setRDim(vector<unsigned int> const &dim)
{
	if(dim.size() != 4) return false;
	if(dim[0] > dim[1] || dim[2] > dim[3]) return false;

	r_dims = dim;
	return true;
}

bool EnvGen::setSDim(vector<unsigned int> const &dim)
{
	if(dim.size() != 4) return false;
	if(dim[0] > dim[1] || dim[2] > dim[3]) return false;

	s_dims = dim;
	return true;
}

bool EnvGen::generateEnv(string const &json_file, string const &img_file)
{
	//Check if the paths for output files are valid ones
	ofstream tmpfile1,tmpfile2;
	tmpfile1.open(json_file);
	if(!tmpfile1.is_open()) return false;
	tmpfile2.open(img_file);
	if(!tmpfile2.is_open() && img_file != "non") return false;
	tmpfile1.close();
	if(tmpfile2.is_open()) tmpfile2.close();

	//Variables to generate unique IDs for states
	unsigned int b_cnt(1);
	unsigned int r_cnt(1);
	unsigned int s_cnt(1);
	unsigned int c_cnt(1);

	//Generate the buildings
	vector<building> env;
	for(unsigned int i = 0; i < n_b; i++)
	{
		building tmp = generateBuilding(b_cnt,r_cnt,s_cnt,c_cnt);
		env.push_back(tmp);
	}

	//Generate the connection between buildings
	vector<building_connection> env_conn = generateConnections(r_cnt,s_cnt,c_cnt,env);

	//Save the environment in a JSON file
	env2Json(json_file,env,env_conn);

	//Create an image to visualize the generated environment
	if(img_file != "non") env2Img(img_file,env,env_conn);

	return true;
}

vector<building_connection> EnvGen::generateConnections(unsigned &r_cnt,unsigned &s_cnt,unsigned &c_cnt, vector<building> const &env)
{
	//Output vector of connections
	vector<building_connection> b_conn;

	//At least a pair of buildings is required in order to generate a connection
	if(env.size() <= 1) return b_conn;

	//The connection components
	vector<int> conn_type;
	vector< vector< vector<string> > > left;
	vector<building> hall;

	//Create a connection between two buildings
	for(unsigned int i = 0; i < env.size()-1; i++)
	{
		//Variables that constitute the i-th connection
		int c_tmp(0);
		vector< vector<string> > l_tmp;
		building h_tmp;

		//Determine if the i-th building will be connected or not wih a hall
		//to the (i+1)-th building
		bool use_hall;
		if(hall_flag == USE_HALLS) use_hall = true;
		else if(hall_flag == DONT_USE_HALLS) use_hall = false;
		else if(hall_flag == MAYBE_USE_HALLS)
		{
			int r = rand() % 100;
			if(r <= (USE_HALL_PROB * 100)) use_hall = true;
			else use_hall = false;
		}

		//Get the i-th building's dimensions
		vector<string> curr_cells = get<3>(env[i]);
		vector<unsigned> curr_dims = get<4>(env[i]);

		//Get the (i+1)-th building's dimensions
		vector<string> next_cells = get<3>(env[i+1]);
		vector<unsigned> next_dims = get<4>(env[i+1]);

		//Get the cells located at the right border of the i-th building's border
		vector<string> r_c;
		unsigned tmp_width = curr_dims[0] * curr_dims[1] * curr_dims[2];
		for(unsigned int j = 0; j < curr_cells.size(); j++)
		{
			//Determine if the j-th cell is located at the building's right-most column
			if((j % tmp_width) == (tmp_width - 1)) r_c.push_back(curr_cells[j]);
		}

		//Get the cells located at the left border of the (i+1)-th building's border
		vector<string> l_c;
		tmp_width = next_dims[0] * next_dims[1] * next_dims[2];
		for(unsigned int j = 0; j < next_cells.size(); j++)
		{
			//Determine if the j-th cell is located at the building's left-most column
			if((j % tmp_width) == 0) l_c.push_back(next_cells[j]);
		}

		//Determine the connection's height
		unsigned min_h = std::min(r_c.size(),l_c.size());
		unsigned h_h = randomPick(h_dims[2],h_dims[3]);
		unsigned s_h = randomPick(s_dims[2],s_dims[3]);
		unsigned conn_h = h_h * s_h;

		//The connection is taller than one (or both) of the buildings
		if(min_h < conn_h)
		{
			//Since the connection height is greater tha both buildings, a door-type
			//connection is used no matter what

			//Set the connection index to indicate that the buildings are
			//connected without a hall
			c_tmp = 1;

			//next b. is taller than the first one
			if(r_c.size() < l_c.size())
			{
				for(unsigned int j = 0; j < r_c.size(); j++) 
				{
					vector<string> ttmp;
					ttmp.push_back(r_c[j]);
					ttmp.push_back(l_c[j]);
					l_tmp.push_back(ttmp);
				}
			}
			//first b. is taller than the next one
			else if(r_c.size() > l_c.size())
			{
				for(unsigned int j = 0; j < l_c.size(); j++) 
				{
					vector<string> ttmp;
					ttmp.push_back(r_c[j]);
					ttmp.push_back(l_c[j]);
					l_tmp.push_back(ttmp);
				}

			}
			// The buildings' height is the same
			else
			{
				for(unsigned int j = 0; j < r_c.size(); j++) 
				{
					vector<string> ttmp;
					ttmp.push_back(r_c[j]);
					ttmp.push_back(l_c[j]);
					l_tmp.push_back(ttmp);
				}
			}
		}
		//Connect this pair of buildings with the specified connection
		else
		{
			//Variable for in case a hall is going to be used
			vector< vector<string> > h_tmp_l;
			vector< vector<string> > h_tmp_a;
			TreeHandle h_tmp_th;
			unsigned h_w = randomPick(h_dims[0],h_dims[1]);//Randomly determine the hall's width dimensions
			unsigned s_w = randomPick(s_dims[0],s_dims[1]);
			vector<string> h_l_c, h_r_c;
			vector<string> s_id,c_id;
			if(use_hall)
			{
				//Generate the unique IDs for the hall's subsections and cells
				string r_id = "r" + to_string(r_cnt++);
				unsigned total_s = h_w*h_h;
				unsigned total_c = h_w * s_w * h_h * s_h;
				for(unsigned int j = 0; j < total_s; j++) s_id.push_back("ss"+to_string(s_cnt++));
				for(unsigned int j = 0; j < total_c; j++) c_id.push_back("c"+to_string(c_cnt++));

				//Generate the 'to-left-of' pairs
				for(unsigned int j = 0; j < (h_h*s_h); j++)
				{
					for(unsigned int k = 0; k < (h_w*s_w - 1); k++)
					{
						unsigned index = j * (h_w*s_w) + k;
						vector<string> tmp;
						tmp.push_back(c_id[index]);
						tmp.push_back(c_id[index + 1]);
						h_tmp_l.push_back(tmp);
					}
				}

				//Generate the 'above' pairs
				for(unsigned int j = 0; j < (h_h*s_h - 1); j++)
				{
					for(unsigned int k = 0; k < (h_w*s_w); k++)
					{
						unsigned index = j * (h_w*s_w) + k;
						vector<string> tmp;
						tmp.push_back(c_id[index]);
						tmp.push_back(c_id[index + (h_w*s_w)]);
						h_tmp_a.push_back(tmp);
					}
				}

				//Generate the hierarchical structure
				h_tmp_th.insert(r_id,0);//This hall is assigned a room-id
				for(unsigned int j = 0; j < s_id.size(); j++) h_tmp_th.insert(r_id,s_id[j],0);
				for(unsigned int j = 0; j < c_id.size(); j++)
				{
					//Store the cells that are located at either the 
					//left or right border of the hall
					if((j % (h_w*s_w)) == 0) h_l_c.push_back(c_id[j]);
					else if((j % (h_w * s_w)) == (h_w * s_w - 1)) h_r_c.push_back(c_id[j]);

					//Compute the cell coordinate
					unsigned row_index = j / (h_w*s_w);
					unsigned col_index = j % (h_w*s_w);

					//Compute the coordinate of the subsection that contains the j-th cell
					unsigned s_row_i = row_index / s_h;
					unsigned s_col_i = col_index / s_w;

					//Tranform the selected subsection coordinate into an index in the 's_id' vector
					unsigned s_index = (s_row_i * h_w) + s_col_i;

					//Insert the j-th cell to its subsection parent child
					h_tmp_th.insert(s_id[s_index],c_id[j],0);
				}
			}

			//Randomly determine the upper row, of each building, to which the connection
			//will be attached
			unsigned lbconn_first_row = randomPick(0, r_c.size() - conn_h);
			unsigned rbconn_first_row = randomPick(0, l_c.size() - conn_h);

			//Attach the hall to both buildings
			if(use_hall)
			{
				//Attach the left-side building to the hall
				for(unsigned int j = 0; j < h_l_c.size(); j++)
				{
					vector<string> tmp;
					tmp.push_back(r_c[lbconn_first_row + j]);
					tmp.push_back(h_l_c[j]);
					h_tmp_l.push_back(tmp);
				}

				//Attach the right-side building to the hall
				for(unsigned int j = 0; j < h_r_c.size(); j++)
				{
					vector<string> tmp;
					tmp.push_back(h_r_c[j]);
					tmp.push_back(l_c[rbconn_first_row + j]);
					h_tmp_l.push_back(tmp);
				}

				//Set the i-th connection
				vector<unsigned> tmp;
				tmp.push_back(1);
				tmp.push_back(h_w);
				tmp.push_back(s_w);
				tmp.push_back(1);
				tmp.push_back(h_h);
				tmp.push_back(s_h);
				h_tmp = building(h_tmp_l,h_tmp_a,h_tmp_th,c_id,tmp);

				//Set the type of connection index for the i-th connection
				c_tmp = 2;
			}
			//Attach buildings directly to each other
			else
			{
				//Set the i-th connection
				for(unsigned int j = 0; j < conn_h; j++)
				{
					vector<string> tmp;
					tmp.push_back(r_c[lbconn_first_row + j]);
					tmp.push_back(l_c[rbconn_first_row + j]);
					l_tmp.push_back(tmp);
				}

				//Set the type of connection index for the i-th connection
				c_tmp = 1;
			}
		}

		//Add the (i)<-->(i+1) pair connection
		conn_type.push_back(c_tmp);
		left.push_back(l_tmp);
		hall.push_back(h_tmp);
	}

	//Build the vector of building connections
	for(unsigned int i = 0; i < conn_type.size(); i++)
	{
		building_connection tmp(conn_type[i],left[i],hall[i]);
		b_conn.push_back(tmp);
	}

	return b_conn;
}

building EnvGen::generateBuilding(unsigned &b_cnt,unsigned &r_cnt,unsigned &s_cnt,unsigned &c_cnt)
{
	//Randomly select the dimensions for the i-th building
	unsigned int b_w,b_h;
	unsigned int r_w,r_h;
	unsigned int s_w,s_h;
	b_w = randomPick(b_dims[0],b_dims[1]);
	b_h = randomPick(b_dims[2],b_dims[3]);
	r_w = randomPick(r_dims[0],r_dims[1]);
	r_h = randomPick(r_dims[2],r_dims[3]);
	s_w = randomPick(s_dims[0],s_dims[1]);
	s_h = randomPick(s_dims[2],s_dims[3]);

	//Generate unique IDs
	unsigned total_rooms = b_w * b_h;
	unsigned total_subsections = total_rooms * r_w * r_h;
	unsigned total_cells = total_subsections * s_w * s_h;
	string b_id = "b" + to_string(b_cnt++);
	vector<string> r_id,s_id,c_id;
	for(unsigned int j = 0; j < total_rooms; j++) r_id.push_back("r"+to_string(r_cnt++));
	for(unsigned int j = 0; j < total_subsections; j++) s_id.push_back("ss"+to_string(s_cnt++));
	for(unsigned int j = 0; j < total_cells; j++) c_id.push_back("c"+to_string(c_cnt++));

	//Compute the amount of cells that will connect rooms vertically and horizontally
	int w_n_conn_cells = ceil(static_cast<float>(r_w)*static_cast<float>(s_w)*ROOM_CONN_RATIO);
	int h_n_conn_cells = ceil(static_cast<float>(r_h)*static_cast<float>(s_h)*ROOM_CONN_RATIO);

	//Initialize the matrices that will store the vertical and horizontal
	//connections between each pair of adjacent rooms
	vector< vector< vector<unsigned> > > ver_conn;
	vector< vector< vector<unsigned> > > hor_conn;
	for(unsigned j = 0; j < (b_h-1); j++)// Init. matrix of vertical connections
	{
		vector< vector<unsigned> > tmp;
		for(unsigned k = 0; k < b_w; k++) tmp.push_back(vector<unsigned>());
		ver_conn.push_back(tmp);
	}
	for(unsigned j = 0; j < b_h; j++)// Init. matrix of horizontal connections
	{
		vector< vector<unsigned> > tmp;
		for(unsigned k = 0; k < (b_w-1); k++) tmp.push_back(vector<unsigned>());
		hor_conn.push_back(tmp);
	}

	//Generate randomly the columns for vertical connection between rooms
	vector<unsigned>::iterator ite;
	for(unsigned int j = 0; j < ver_conn.size(); j++)
	{
		for(unsigned int k = 0; k < ver_conn[j].size(); k++)
		{
			//Generate 'w_n_conn_cells' vertical connections in each pair of rooms
			for(int l = 0; l < w_n_conn_cells; l++)
			{
				while(true)
				{
					//Random and non-repeated columns
					unsigned tmp = randomPick(0,s_w*r_w-1);
					ite = find(ver_conn[j][k].begin(),ver_conn[j][k].end(),tmp);
					if(ite == ver_conn[j][k].end())
					{
						ver_conn[j][k].push_back(tmp);
						break;
					}
				}
			}
		}
	}
	//Generate randomly the rows for horizontal connection between rooms
	for(unsigned int j = 0; j < hor_conn.size(); j++)
	{
		for(unsigned int k = 0; k < hor_conn[j].size(); k++)
		{
			//Generate 'h_n_conn_cells' horizontal connections in each pair of rooms
			for(int l = 0; l < h_n_conn_cells; l++)
			{
				while(true)
				{
					//Random and non-repeated columns
					unsigned tmp = randomPick(0,s_h*r_h-1);
					ite = find(hor_conn[j][k].begin(),hor_conn[j][k].end(),tmp);
					if(ite == hor_conn[j][k].end())
					{
						hor_conn[j][k].push_back(tmp);
						break;
					}
				}
			}
		}
	}

	//Generate the grid of cells with full connectivity,
	//with exception of the walls between rooms
	vector< vector<string> > left;
	vector< vector<string> > above;
	unsigned b_w_c = b_w * r_w * s_w;
	unsigned b_h_c = b_h * r_h * s_h;

	//Generate horizontal connectivity pairs
	for(unsigned int j = 0; j < b_h_c; j++)
	{
		for(unsigned int k = 0; k < (b_w_c - 1); k++)
		{
			//Index in the 'c_id' cells vector
			int index = j*b_w_c + k;

			//Determine if the current cell is adjacent to another room
			bool add_pair(true);
			if(((k+1) % (r_w * s_w)) == 0)
			{
				//Get the indexes in the matrix of horizontal connectivity vectors
				int row_i = j / (r_h * s_h);
				int col_i = k / (r_w * s_w);

				//Map the current row index in the cell grid to a one considering 
				//only the rows in the current room
				unsigned loc_row = j - (row_i * r_h * s_h);

				//Look for this row in the list of no-connectivity rows
				ite = find(hor_conn[row_i][col_i].begin(),hor_conn[row_i][col_i].end(),loc_row);
				if(ite == hor_conn[row_i][col_i].end())
				{
					add_pair = false;
				}
			}

			//If there is not a wall between the current cell and the next one
			//add the connectivity pair
			if(add_pair)
			{
				vector<string> tmp;
				tmp.push_back(c_id[index]);
				tmp.push_back(c_id[index+1]);
				left.push_back(tmp);
			}
		}
	}

	//Generate vertical connectivity pairs
	for(unsigned int j = 0; j < (b_h_c - 1); j++)
	{
		for(unsigned int k = 0; k < b_w_c; k++)
		{
			//Index in the 'c_id' cells vector
			int index = j*b_w_c + k;

			//Determine if the current cell is adjacent to another room
			bool add_pair(true);
			if(((j+1) % (r_h * s_h)) == 0)
			{
				//Get the indexes in the matrix of horizontal connectivity vectors
				int row_i = j / (r_h * s_h);
				int col_i = k / (r_w * s_w);

				//Map the current column index in the cell grid to a one considering 
				//only the columns in the current room
				unsigned loc_col = k - (col_i * r_w * s_w);

				//Look for this column in the list of no-connectivity columns
				ite = find(ver_conn[row_i][col_i].begin(),ver_conn[row_i][col_i].end(),loc_col);
				if(ite == ver_conn[row_i][col_i].end())
				{
					add_pair = false;
				}
			}

			//If there is not a wall between the current cell and the next one
			//add the connectivity pair
			if(add_pair)
			{
				vector<string> tmp;
				tmp.push_back(c_id[index]);
				tmp.push_back(c_id[index + b_w_c]);
				above.push_back(tmp);
			}
		}
	}

	//Generate the hierarchical structure
	TreeHandle th;

	//Insert the building
	th.insert(b_id,0);

	//Insert the rooms
	for(unsigned int i = 0; i < r_id.size(); i++) th.insert(b_id,r_id[i],0);

	//Insert subsections
	unsigned b_w_s = b_w * r_w;
	for(unsigned int i = 0; i < s_id.size(); i++)
	{
		//Transform the subsection index into a (row, col) coordinate
		unsigned subsec_row = i / b_w_s;
		unsigned subsec_col = i % b_w_s;

		//Compute the room coordinate to which this subsection belongs to
		unsigned room_row = subsec_row / r_h;
		unsigned room_col = subsec_col / r_w;

		//Transform the room coordinate into an index in 'r_id'
		unsigned room_index = room_row * b_w + room_col;

		//Insert the i-th subsection as child node of the computed room
		th.insert(r_id[room_index], s_id[i], 0);
	}

	//Insert cells
	for(unsigned int i = 0; i < c_id.size(); i++)
	{
		//Transform the cell index into a (row, col) coordinate
		unsigned cell_row = i / b_w_c;
		unsigned cell_col = i % b_w_c;

		//Compute the subsection coordinate to which this cell belongs to
		unsigned subsec_row = cell_row / s_h;
		unsigned subsec_col = cell_col / s_w;

		//Transform the subsection coordinate into an index in 's_id'
		unsigned subsec_index = subsec_row * (b_w * r_w) + subsec_col;

		//Insert the i-th cell as child node of the computed subsection
		th.insert(s_id[subsec_index], c_id[i], 0);
	}

	//Save the building's dimensions for later be used to generate
	//a connection with another building
	vector<unsigned> b_full_dim;
	b_full_dim.push_back(b_w);
	b_full_dim.push_back(r_w);
	b_full_dim.push_back(s_w);
	b_full_dim.push_back(b_h);
	b_full_dim.push_back(r_h);
	b_full_dim.push_back(s_h);

	return building(left,above,th,c_id,b_full_dim);
}

bool EnvGen::env2Img(string const &img_file,vector<building> const &env,vector<building_connection> const &env_conn)
{
	//Check for consistency of the input params
	if(env.size() != (env_conn.size() + 1)) return false;

	//Define the cell's (which are squared) dimensions pixels
	int c = cell_pixel_dim;

	//Compute the image's width
	int i_w(0);
	for(unsigned int i = 0; i < env.size(); i++)//Add the buildings' width in cells
	{
		vector<unsigned> dims = get<4>(env[i]);
		unsigned b_width = dims[0] * dims[1] * dims[2];
		i_w += static_cast<int>(b_width);
	}
	for(unsigned int i = 0; i < env_conn.size(); i++)//Add the halls' width in cells
	{
		//The i-th connection is a hall
		if(get<0>(env_conn[i]) == 2)
		{
			building tmp = get<2>(env_conn[i]);
			vector<unsigned> dims = get<4>(tmp);
			unsigned h_width = dims[0] * dims[1] * dims[2];
			i_w += static_cast<int>(h_width);
		}
	}

	//Compute the image's height
	int upp_dist(0);
	int low_dist(bHeight(env[0]));
	int offset(0);
	vector<int> b_ori;
	vector<int> c_ori;
	b_ori.push_back(0);//Add the first building's origin
	for(unsigned int i = 0; i < env_conn.size(); i++)
	{
		//Get the list of cells & dimensions of the buildings connected by the i-th connection
		vector<string> lbc_vec = get<3>(env[i]);
		vector<string> rbc_vec = get<3>(env[i+1]);
		vector<unsigned> lb_dim = get<4>(env[i]);
		vector<unsigned> rb_dim = get<4>(env[i+1]);
		string lbc;
		string rbc;
		int lbc_row;
		int rbc_row;
		int lb_h;
		int rb_h;

		//Direct connection
		if(get<0>(env_conn[i]) == 1)
		{
			//Get the first pair of cells that connect these buildings
			vector<vector<string> > tmp = get<1>(env_conn[i]);
			lbc = tmp[0][0];
			rbc = tmp[0][1];
		}
		//Hall connection
		else
		{
			//Get first row of each building, at which they connect to the hall
			building tmp = get<2>(env_conn[i]);
			vector<string> h_cell = get<3>(tmp);
			vector<vector<string> > h_left = get<0>(tmp);
			vector<unsigned> h_d = get<4>(tmp);
			string hcl = h_cell[0];
			string hcr = h_cell[h_d[0] * h_d[1] * h_d[2] - 1];

			bool got_l(false);
			bool got_r(false);
			for(unsigned int j = 0; j < h_left.size(); j++)
			{
				//Found the pair containing the left building's cell
				if(h_left[j][1] == hcl)
				{
					lbc = h_left[j][0];
					got_l = true;
				}

				//Found the pair containing the right building's cell
				if(h_left[j][0] == hcr)
				{
					rbc = h_left[j][1];
					got_r = true;
				}

				if(got_l && got_r) break;
			}
		}

		//Get the row at which each of these cells are located in their respective buildings
		vector<string>::iterator ite = find(lbc_vec.begin(),lbc_vec.end(),lbc);
		int index = std::distance(lbc_vec.begin(),ite);
		lbc_row = index / (lb_dim[0]*lb_dim[1]*lb_dim[2]);
		ite = find(rbc_vec.begin(),rbc_vec.end(),rbc);
		index = std::distance(rbc_vec.begin(),ite);
		rbc_row = index / (rb_dim[0]*rb_dim[1]*rb_dim[2]);

		//Compute each buildings' height
		lb_h = bHeight(env[i]);
		rb_h = bHeight(env[i+1]);

		//-------------------------------------------------------------

		//Update the offset
		offset += (rbc_row - lbc_row);

		//Update the upper distance from the first building's origin
		upp_dist = std::max(upp_dist,offset);

		//Update the lower distance from the first building's origin
		low_dist = std::max(low_dist,(rb_h - offset));

		//Save right-building origin row
		b_ori.push_back(offset);

		//Save the i-th connection's origin row (w.r.t. the left building)
		c_ori.push_back(lbc_row);
	}

	//Compute the image's height
	int i_h = upp_dist + low_dist;

	//Update the buildings' origin row
	for(unsigned int i = 0; i < b_ori.size(); i++) b_ori[i] = (upp_dist - b_ori[i]);

	//Create the canvas image
	Mat img(i_h*c,i_w*c,CV_8UC3,Scalar(0,0,0));

	//Create a JSON file with the coordinate of each cell in the image
	json cell_coor;
	cell_coor["cell_dim"] = cell_pixel_dim;
	cell_coor["data"] = json::array();

	//Draw buildings & halls
	unsigned x_offset(0);
	for(unsigned int i = 0; i < env.size(); i++)
	{
		//Define the i-th building's dimensions in pixels
		vector<string> cells = get<3>(env[i]);
		vector<unsigned> dims = get<4>(env[i]);
		unsigned b_w = dims[0] * dims[1] * dims[2];
		unsigned b_h = (cells.size() / b_w) * c;
		b_w *= c;
		Rect b_rec(x_offset, b_ori[i]*c, b_w, b_h);

		//Draw the building as a white box
		rectangle(img,b_rec,Scalar(255,255,255),-1);

		//Draw the building's cell-IDs
		unsigned bc_w = dims[0] * dims[1] * dims[2];
		Scalar t_color(0,0,0);
		for(unsigned int j = 0; j < cells.size(); j++)
		{
			//Compute j-th cell's row & column
			unsigned c_row = j / bc_w;
			unsigned c_col = j % bc_w;

			//Compute the text's origin in pixels
			unsigned x_ori = x_offset + (c_col * cell_pixel_dim);
			unsigned y_ori = (b_ori[i]*c) + (c_row * cell_pixel_dim) + (cell_pixel_dim / 2);
			Point cell_p(x_ori,y_ori);

			//Draw the cell ID
			putText(img,cells[j],cell_p,FONT_HERSHEY_PLAIN,1.0,t_color,1);

			//Add the cell's center coordinate
			json cc_tmp;
			cc_tmp["id"] = cells[j];
			cc_tmp["x"] = x_ori + c/2;
			cc_tmp["y"] = y_ori;
			cell_coor["data"].push_back(cc_tmp);
		}

		//Update the x-offset
		x_offset += b_w;

		//Check if the i-th connection is a hall, in order to draw it
		if(i != (env.size() - 1))
		{
			if(get<0>(env_conn[i]) == 2)
			{
				//Define the i-th connection's (which is a hall) dimensions in pixels
				building tmp = get<2>(env_conn[i]);
				cells = get<3>(tmp);
				dims = get<4>(tmp);
				unsigned h_w = dims[0] * dims[1] * dims[2];
				unsigned h_h = (cells.size() / h_w) * c;
				h_w *= c;
				Rect h_rec(x_offset, (b_ori[i] + c_ori[i])*c, h_w, h_h);

				//Draw the building as a white box
				rectangle(img,h_rec,Scalar(255,255,255),-1);

				//Draw the building's cell-IDs
				unsigned hc_w = dims[0] * dims[1] * dims[2];
				Scalar t_color(0,0,0);
				for(unsigned int j = 0; j < cells.size(); j++)
				{
					//Compute j-th cell's row & column
					unsigned c_row = j / hc_w;
					unsigned c_col = j % hc_w;

					//Compute the text's origin in pixels
					unsigned x_ori = x_offset + (c_col * cell_pixel_dim);
					unsigned y_ori = ((b_ori[i] + c_ori[i])*c) + (c_row * cell_pixel_dim) + (cell_pixel_dim / 2);
					Point cell_p(x_ori,y_ori);

					//Draw the cell ID
					putText(img,cells[j],cell_p,FONT_HERSHEY_PLAIN,1.0,t_color,1);
				}

				//Update the x-offset
				x_offset += h_w;
			}
		}
	}

	//Save the cell-coord. JSON file
	vector<string> tokens = splitStr(img_file,"/");
	string cc_fname("");
	if(img_file[0] == '/') cc_fname += "/";
	for(unsigned int i = 0; i < (tokens.size()-1); i++)
	{
		cc_fname += (tokens[i]+"/");
	}
	cc_fname += "cell_coord.json";
	ofstream cc_fw(cc_fname);
	cc_fw << cell_coor.dump(4);
	cc_fw.close();

	//Get a copy of the black and white image
	Mat cp_img = img.clone();

	//++++++++++++ Draw cells, subsections and rooms divisions ++++++++++++
	//Draw cell lines
	for(int i = 0; i < img.cols; i++)// Vertical lines
	{
		if(i == 0 || (i % c) == (c - 1))
		{
			line(img,Point(i,0),Point(i,img.rows-1),Scalar(255,0,0),2);
		}
	}
	for(int i = 0; i < img.rows; i++)// Vertical lines
	{
		if(i == 0 || (i % c) == (c - 1))
		{
			line(img,Point(0,i),Point(img.cols-1,i),Scalar(255,0,0),2);
		}
	}

	//Vector of X-origin for each building (in pixels)
	vector<int> bx_ori;

	//Draw subsection lines
	x_offset = 0;
	for(unsigned int i = 0; i < env.size(); i++)
	{
		//Save the i-th building's X origin
		bx_ori.push_back(x_offset);

		//Draw for the i-th building
		vector<unsigned> dims = get<4>(env[i]);
		int bui_w = dims[0] * dims[1] * dims[2] * c;
		int bui_h = dims[3] * dims[4] * dims[5] * c;
		int sub_w = dims[2] * c;
		int sub_h = dims[5] * c;
		for(int j = x_offset; j < (x_offset + bui_w); j++)// Vertical lines
		{
			if(j == 0 || (j % (sub_w)) == (sub_w - 1))
			{
				line(img,Point(j,0),Point(j,img.rows-1),Scalar(0,255,0),2);
			}
		}
		for(int j = b_ori[i]*c; j < img.rows; j++)// Horizontal lines
		{
			if(j == b_ori[i]*c || ((j-b_ori[i]*c) % (sub_h)) == (sub_h - 1))
			{
				line(img,Point(x_offset,j),Point(x_offset+bui_w-1,j),Scalar(0,255,0),2);
			}
		}
		x_offset += bui_w;

		//Draw for the i-th connection if it is a hall
		//if(env_conn.size() > 0)
		if(i != env.size()-1)
		{
			if(get<0>(env_conn[i]) == 2)
			{
				building tmp = get<2>(env_conn[i]);
				dims = get<4>(tmp);
				bui_w = dims[0] * dims[1] * dims[2] * c;
				bui_h = dims[3] * dims[4] * dims[5] * c;
				sub_w = dims[2] * c;
				sub_h = dims[5] * c;
				for(int j = x_offset; j < (x_offset + bui_w); j++)// Vertical lines
				{
					if(j == 0 || (j % sub_w) == (sub_w - 1))
					{
						line(img,Point(j,0),Point(j,img.rows-1),Scalar(0,255,0),2);
					}
				}
				for(int j = (b_ori[i]+c_ori[i])*c; j < img.rows; j++)// Horizontal lines
				{
					if(j == (b_ori[i]+c_ori[i])*c || ((j-(b_ori[i]+c_ori[i])*c) % (sub_h)) == (sub_h - 1))
					{
						line(img,Point(x_offset,j),Point(x_offset+bui_w-1,j),Scalar(0,255,0),2);
					}
				}
				x_offset += bui_w;
			}
		}


	}

	//Draw the room dividing segments
	for(unsigned int i = 0; i < env.size(); i++)
	{
		//Info from the building necessary to find those pair of 
		//cells that are not neighbors
		vector< vector<string> > left = get<0>(env[i]);
		vector< vector<string> > above = get<1>(env[i]);
		vector<string> cells = get<3>(env[i]);
		vector<unsigned> dims = get<4>(env[i]);

		//Vectors of cell indexes without neighbor to their right & below
		vector< vector<unsigned> > non_neig_r;
		vector< vector<unsigned> > non_neig_b;

		//Get the cells with no neighbor at their right, that are not border cells
		for(unsigned j = 0; j < (dims[0] - 1); j++)
		{
			//Compute the cell-column at the right of the current wall
			unsigned curr_col = ((j+1) * dims[1] * dims[2]) - 1;

			//Iterate over all the building's cells & if those located 
			//at the current column do not have a neighbor to its right
			for(unsigned int k = 0; k < cells.size(); k++)
			{
				//Compute the k-th cell's column
				unsigned k_col = k % (dims[0] * dims[1] * dims[2]);

				//The k-th cell is in the current column
				if(k_col == curr_col)
				{
					//Check if the k-th cell is neighbor of the cell at its right
					vector<string> tmp;
					tmp.push_back(cells[k]);
					tmp.push_back(cells[k+1]);
					vector< vector<string> >::iterator ite;
					ite = find(left.begin(),left.end(),tmp);

					// k and k+1 are divided by a wall segment
					if(ite == left.end())
					{
						//Save the coordinate of the k-th cell
						vector<unsigned> coor;
						coor.push_back(k / (dims[0] * dims[1] * dims[2]));//cell's row
						coor.push_back(k_col);//cell's col
						non_neig_r.push_back(coor);
					}
				}
			}
		}

		//Get the cells with no neighbor below it, that are not border cells
		for(unsigned j = 0; j < (dims[3] - 1); j++)
		{
			//Compute the cell-row above the current wall
			unsigned curr_row = ((j+1) * dims[4] * dims[5]) - 1;

			//Iterate over all the building's cells & if those located 
			//at the current row do not have a neighbor below it
			for(unsigned int k = 0; k < cells.size(); k++)
			{
				//Compute the k-th cell's row
				unsigned k_row = k / (dims[0] * dims[1] * dims[2]);

				//The k-th cell is in the current row
				if(k_row == curr_row)
				{
					//Check if the k-th cell is neighbor of the cell at its right
					vector<string> tmp;
					tmp.push_back(cells[k]);
					tmp.push_back(cells[k + (dims[0] * dims[1] * dims[2])]);
					vector< vector<string> >::iterator ite;
					ite = find(above.begin(),above.end(),tmp);

					// k and k+(building-width) are divided by a wall segment
					if(ite == above.end())
					{
						//Save the coordinate of the k-th cell
						vector<unsigned> coor;
						coor.push_back(k_row);//cell's row
						coor.push_back(k % (dims[0] * dims[1] * dims[2]));//cell's col
						non_neig_b.push_back(coor);
					}
				}
			}
		}

		//Draw the segments that correspond to dividing walls
		// Vertical segments
		for(unsigned int j = 0; j < non_neig_r.size(); j++)
		{
			//Compute the start & end points
			int x_coor = bx_ori[i] + (non_neig_r[j][1] + 1) * c;
			int y_coor1 = (b_ori[i] + non_neig_r[j][0]) * c;
			int y_coor2 = y_coor1 + c;
			Point p1(x_coor,y_coor1);
			Point p2(x_coor,y_coor2);

			//Draw the segement
			line(img,p1,p2,Scalar(0,0,0),3);
		}
		// Horizontal segments
		for(unsigned int j = 0; j < non_neig_b.size(); j++)
		{
			//Compute the start & end points
			int x_coor1 = bx_ori[i] + non_neig_b[j][1] * c;
			int x_coor2 = x_coor1 + c;
			int y_coor = (b_ori[i] + non_neig_b[j][0] + 1) * c;
			Point p1(x_coor1,y_coor);
			Point p2(x_coor2,y_coor);

			//Draw the segement
			line(img,p1,p2,Scalar(0,0,0),3);
		}
	}

	//Draw a black line between adjacent buildings that are connected
	//by a door, without over the door-connection
	for(unsigned int i = 0; i < env_conn.size(); i++)
	{
		if(get<0>(env_conn[i]) == 1)
		{
			//Get cells in the left building located at the top 
			//& bottom of the connection
			vector<vector<string> > conn = get<1>(env_conn[i]);
			string top_lc = conn[0][0];
			string bot_lc = conn[conn.size()-1][0];

			//Get these cells' rows
			vector<unsigned> dims = get<4>(env[i]);
			vector<string> lb_cells = get<3>(env[i]);
			int top_row(-1);
			int bot_row(-1);
			for(unsigned int j = 0; j < lb_cells.size(); j++)
			{
				//Compute the top-cell row
				if(lb_cells[j] == top_lc) top_row = j / (dims[0]*dims[1]*dims[2]);

				//Compute the bottom-cell row
				if(lb_cells[j] == bot_lc) bot_row = j / (dims[0]*dims[1]*dims[2]);

				if(top_row > -1 && bot_row > -1) break;
			}

			//Compute the top & bottom coordinates of the connection
			int x_coor = bx_ori[i+1];
			int y_top_coor = (b_ori[i] + top_row)*c;
			int y_bot_coor = (b_ori[i] + bot_row + 1)*c;

			//Draw the upper & lower segments
			line(img,Point(x_coor,y_top_coor),Point(x_coor,0),Scalar(0),3);//upper
			line(img,Point(x_coor,y_bot_coor),Point(x_coor,img.rows-1),Scalar(0),3);//upper
		}
	}

	//Erase the division segments that are out of a room
	for(int j = 0; j < cp_img.rows; j++)
	{
		for(int i = 0; i < cp_img.cols; i++)
		{
			Vec3b intensity = cp_img.at<Vec3b>(j,i);
			if(intensity.val[0] == 0)
			{
				intensity.val[0] = 0;
				intensity.val[1] = 0;
				intensity.val[2] = 0;
				img.at<Vec3b>(j,i) = intensity;
			}
		}
	}

	//Save the image file
	imwrite(img_file,img);

	return true;
}

bool EnvGen::env2Json(string const &json_file,vector<building> const &env,vector<building_connection> const &env_conn)
{
	//Check for consistency of the input params
	if(env.size() != (env_conn.size() + 1)) return false;

	json jj;
	jj["Buildings"] = json::array();
	jj["to_left_of"] = json::array();
	jj["above"] = json::array();

	//Write Buildings & Halls as rooms ------------------------------------------------
	for(unsigned int i = 0; i < env.size(); i++)
	{
		//Building
		vector<string> tmp;
		TreeHandle th = get<2>(env[i]);
		tmp = th.keysAtLevel(1);
		string b_name = tmp[0];
		jj["Buildings"].push_back({{"Name",b_name}});

		//Get the index of the current building
		int index(-1);
		for(unsigned int j = 0; j < jj["Buildings"].size(); j++)
		{
			if(jj["Buildings"][j]["Name"] == b_name)
			{
				index = j;
				break;
			}
		}
		jj["Buildings"][index]["Rooms"] = json::array();

		//Insert the building's dimensions
		jj["Buildings"][index]["Dims"] = json::array();
		vector<unsigned> tmp_dim = get<4>(env[i]);
		for(unsigned int j = 0; j < tmp_dim.size(); j++) jj["Buildings"][index]["Dims"].push_back(tmp_dim[j]);

		//Rooms
		tmp = th.keysOfChildren(b_name);
		for(unsigned int j = 0; j < tmp.size(); j++)
		{
			jj["Buildings"][index]["Rooms"].push_back({{"Name",tmp[j]}});

			//Get the index of the current room
			int rindex(-1);
			for(unsigned int k = 0; k < jj["Buildings"][index]["Rooms"].size(); k++)
			{
				if(jj["Buildings"][index]["Rooms"][k]["Name"] == tmp[j])
				{
					rindex = k;
					break;
				}
			}
			jj["Buildings"][index]["Rooms"][rindex]["Subsections"] = json::array();

			//Flag to difference from halls
			jj["Buildings"][index]["Rooms"][rindex]["is_hall"] = false;

			//Subsections
			vector<string> subsec = th.keysOfChildren(tmp[j]);
			for(unsigned int k = 0; k < subsec.size(); k++)
			{
				jj["Buildings"][index]["Rooms"][rindex]["Subsections"].push_back({{"Name",subsec[k]}});

				//Get the index of  the current subsection
				int sindex(-1);
				for(unsigned int l = 0; l < jj["Buildings"][index]["Rooms"][rindex]["Subsections"].size(); l++)
				{
					if(jj["Buildings"][index]["Rooms"][rindex]["Subsections"][l]["Name"] == subsec[k])
					{
						sindex = l;
						break;
					}
				}
				jj["Buildings"][index]["Rooms"][rindex]["Subsections"][sindex]["Cells"] = json::array();

				//Cells
				vector<string> cell = th.keysOfChildren(subsec[k]);
				for(unsigned int l = 0; l < cell.size(); l++)
				{
					jj["Buildings"][index]["Rooms"][rindex]["Subsections"][sindex]["Cells"].push_back({{"Name",cell[l]}});
				}

			}
		}

		//Hall
		if(i < env_conn.size())
		{
			//the i-th connection is a hall
			if(get<0>(env_conn[i]) == 2)
			{
				//Get the TreeHandle of the hall
				building hall_b = get<2>(env_conn[i]);
				th = get<2>(hall_b);
				tmp = th.keysAtLevel(1);
				jj["Buildings"][index]["Rooms"].push_back({{"Name",tmp[0]}});

				//Get the hall's room index
				int rindex;
				for(unsigned int j = 0; j < jj["Buildings"][index]["Rooms"].size(); j++)
				{
					if(jj["Buildings"][index]["Rooms"][j]["Name"] == tmp[0])
					{
						rindex = j;
						break;
					}
				}
				jj["Buildings"][index]["Rooms"][rindex]["Subsections"] = json::array();

				//Flag to identify it as a hall & insert its dimensions
				jj["Buildings"][index]["Rooms"][rindex]["is_hall"] = true;
				jj["Buildings"][index]["Rooms"][rindex]["Dims"] = json::array();
				vector<unsigned> tmp_dim = get<4>(hall_b);
				for(unsigned int j = 0; j < tmp_dim.size(); j++) jj["Buildings"][index]["Rooms"][rindex]["Dims"].push_back(tmp_dim[j]);

				//Subsections
				vector<string> tmp_s = th.keysOfChildren(tmp[0]);
				for(unsigned int j = 0; j < tmp_s.size(); j++)
				{
					jj["Buildings"][index]["Rooms"][rindex]["Subsections"].push_back({{"Name",tmp_s[j]}});

					//Get the index of the current subsection
					int sindex;
					for(unsigned int k = 0; k < jj["Buildings"][index]["Rooms"][rindex]["Subsections"].size(); k++)
					{
						if(jj["Buildings"][index]["Rooms"][rindex]["Subsections"][k]["Name"] == tmp_s[j])
						{
							sindex = k;
							break;
						}
					}
					jj["Buildings"][index]["Rooms"][rindex]["Subsections"][sindex]["Cells"] = json::array();

					//Cells
					vector<string> tmp_cell = th.keysOfChildren(tmp_s[j]);
					for(unsigned int k = 0; k < tmp_cell.size(); k++)
					{
						jj["Buildings"][index]["Rooms"][rindex]["Subsections"][sindex]["Cells"].push_back({{"Name",tmp_cell[k]}});
					}
				}
			}
		}
	}

	//Write Buildings & Halls' left & above pairs -------------------------------------
	for(unsigned int i = 0; i < env.size(); i++)
	{
		//i-th building's pairs
		vector< vector<string> > lp, ap;
		lp = get<0>(env[i]);
		ap = get<1>(env[i]);
		for(unsigned int j = 0; j < lp.size(); j++)
		{
			jj["to_left_of"].push_back(
			{
				{"subject",lp[j][0]},
				{"reference",lp[j][1]}
			});
		}
		for(unsigned int j = 0; j < ap.size(); j++)
		{
			jj["above"].push_back(
			{
				{"subject",ap[j][0]},
				{"reference",ap[j][1]}
			});
		}

		//i-th connection's pairs
		if(i < env_conn.size())
		{
			//Door connection
			if(get<0>(env_conn[i]) == 1)
			{
				lp = get<1>(env_conn[i]);
				for(unsigned int j = 0; j < lp.size(); j++)
				{

					jj["to_left_of"].push_back(
					{
						{"subject",lp[j][0]},
						{"reference",lp[j][1]}
					});
				}
			}
			//Hall connection
			else if(get<0>(env_conn[i]) == 2)
			{
				building hall_b = get<2>(env_conn[i]);
				lp = get<0>(hall_b);
				ap = get<1>(hall_b);
				for(unsigned int j = 0; j < lp.size(); j++)
				{

					jj["to_left_of"].push_back(
					{
						{"subject",lp[j][0]},
						{"reference",lp[j][1]}
					});
				}
				for(unsigned int j = 0; j < ap.size(); j++)
				{

					jj["above"].push_back(
					{
						{"subject",ap[j][0]},
						{"reference",ap[j][1]}
					});
				}
			}
		}
	}

	ofstream jfile;
	jfile.open(json_file);
	jfile << jj.dump(4);
	jfile.close();

	return true;
}

vector<std::string> EnvGen::splitStr(std::string const &fullString,std::string const &delimiter)
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
