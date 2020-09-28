/**
 * \class EnvGen
 * 
 * \brief This class is for generating navigation environments with a description of spatial neighborhood relations.
 * 
 * \author $Author: Sergio A. Serrano$
 * 
 * \date $Date: 25/04/19$
 * 
 * Contact: sserrano@inaoep.mx
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <exception>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <TreeHandle.hpp>

using namespace std;

#ifndef ENV_GEN
#define ENV_GEN

//building contains:
// - Left relation pairs
// - Above relation pairs
// - Hierarchical structure of the building
// - List of cells
// - Vector with the building's dimensions: b_w, r_w, s_w, b_h, r_h, s_h
typedef tuple<vector<vector<string> >, vector<vector<string> >, TreeHandle, vector<string>, vector<unsigned> > building;

//building_connection conatins:
// - index of the type of connection used: 1 for door & 2 for hall
// - left relation pairs (for door connection)
// - building for representing a hall connection
typedef tuple<int, vector<vector<string> >, building> building_connection;

class EnvGen
{
	public:

		const static unsigned int USE_HALLS = 0;
		const static unsigned int DONT_USE_HALLS = 1;
		const static unsigned int MAYBE_USE_HALLS = 2;

	private:

		//Probability of connecting a pair of buildings when the 'hall_flag'
		//is set to MAYBE_USE_HALLS
		float USE_HALL_PROB;

		//Porcentage of connection between two adjacent rooms
		float ROOM_CONN_RATIO;

		//The width & height in pixels for drawing cells
		int cell_pixel_dim;

		unsigned int n_b;//Amount of buildings
		unsigned int hall_flag;//Whether use halls to connect buildings or not, or decide it randomly
		vector<unsigned int> b_dims;//in terms of rooms
		vector<unsigned int> h_dims;//in terms of subsections
		vector<unsigned int> r_dims;//in terms of subsections
		vector<unsigned int> s_dims;//in terms of cells

		/**
		 * \brief Method for loading the full set of configuration parameters from a JSON file.
		 * \param config_file Name of the JSON file containing the configuration parameters.
		 * \return Returns true if the file was successfully loaded, otherwise false.
		*/
		bool loadConfig(string const &config_file);

		/**
		 * \brief Method for selecting pseudo-randomly an integer from a closed interval.
		 * \param l_bound Lower bound of the interval.
		 * \param u_bound Upper bound of the interval.
		 * \return Returns the selected integer.
		*/
		unsigned int randomPick(unsigned int const &l_bound, unsigned int const &u_bound);

		/**
		 * \brief Method for computing the height (in cells) of a non-empty building
		 * \param b Building whose height will be computed.
		 * \return Returns the buildings height.
		*/
		int bHeight(building const &b);

		/**
		 * \brief Method for generating a building structure.
		 * \param b_cnt Reference to the index generator variable for buildings.
		 * \param r_cnt Reference to the index generator variable for rooms.
		 * \param s_cnt Reference to the index generator variable for sub-sections.
		 * \param c_cnt Reference to the index generator variable for cells.
		 * \return Returns the resulting 'building' structure.
		*/
		building generateBuilding(unsigned &b_cnt,unsigned &r_cnt,unsigned &s_cnt,unsigned &c_cnt);

		/**
		 * \brief This method is for creating connections between continuous buildings, such connections might be in the form of a hall, or as door that directly connects a pair of buildings. For a vector with N buildings, (N-1) connections will be created. When a hall is used to connect a pair of buildings, such hall will be integrated as a room of the building at its left.
		 * \param r_cnt Reference to the index generator variable for rooms.
		 * \param s_cnt Reference to the index generator variable for sub-sections.
		 * \param c_cnt Reference to the index generator variable for cells.
		 * \param env Vector of buildings
		 * \return Returns a vector of building-connections.
		*/
		vector<building_connection> generateConnections(unsigned &r_cnt,unsigned &s_cnt,unsigned &c_cnt, vector<building> const &env);

		/**
		 * \brief Method for writing a JSON file that contains the hierarchical structure & spatial relation pairs of the environment described by the buildings and connections provided as input parameters.
		 * \param json_file Name of the output JSON file.
		 * \param env Vector of buildings.
		 * \param env_conn Vector of connections for the building in 'env'.
		 * \return Returns true if the JSON was successfully created, otherwise false.
		*/
		bool env2Json(string const &json_file,vector<building> const &env,vector<building_connection> const &env_conn);

		/**
		 * \brief Method for rendering an image of the environment described by the set of buildings and connections provided as input parameters.
		 * \param img_file Name of the output image file.
		 * \param env Vector of buildings.
		 * \param env_conn Vector of connections for the building in 'env'.
		 * \return Returns true if the image was successfully created, otherwise false.
		*/
		bool env2Img(string const &img_file,vector<building> const &env,vector<building_connection> const &env_conn);

	public:

		/**
		 * \brief Default constructor that initializes with default parameters.
		*/
		EnvGen();

		/**
		 * \brief Constructor that initializes by loading a JSON file that must contain two positive integer fields (n_b and hall_flag) and four vector fields (b_dims,h_dims,r_dims,s_dims), each with four positive integers. In each vector, the first two values are a closed interval of posible width value, while the last two are for the height.
		 * \param config_file Name of the JSON file containing the configuration parameters.
		*/
		EnvGen(string const &config_file);

		/**
		 * \brief Get method for the amount of buildings to be generated.
		 * \return Returns 'n_b'.
		*/
		unsigned int getNB() const;

		/**
		 * \brief Get method for whether buildings are going to be connected with halls.
		 * \return Returns 'hall_flag'.
		*/
		unsigned int getHF() const;

		/**
		 * \brief Get method for the ranges of possible values of width and height (in terms of rooms) for buildings.
		 * \return Returns 'b_dims'.
		*/
		vector<unsigned int> getBDim() const;

		/**
		 * \brief Get method for the ranges of possible values of width and height (in terms of subsections) for halls.
		 * \return Returns 'h_dims'.
		*/
		vector<unsigned int> getHDim() const;

		/**
		 * \brief Get method for the ranges of possible values of width and height (in terms of subsections) for rooms.
		 * \return Returns 'r_dims'.
		*/
		vector<unsigned int> getRDim() const;

		/**
		 * \brief Get method for the ranges of possible values of width and height (in terms of cells) for subsections.
		 * \return Returns 's_dims'.
		*/
		vector<unsigned int> getSDim() const;

		/**
		 * \brief Set method for the amount of buildings to be generated.
		 * \param n_b Amount of buildings to be generated.
		*/
		void setNB(unsigned int n_b);

		/**
		 * \brief Set method for the flag of whether buildings will be connected by halls.
		 * \param hall_flag New value for the hall-connection flag
		 * \return Returns true if the assignation was successfull, otherwise false.
		*/
		bool setHF(unsigned int hall_flag);

		/**
		 * \brief Set method for buildings possible dimensions.
		 * \param dim New pair of intervals of possible dimensions.
		 * \return Returns true if the assignation was successfull, otherwise false.
		*/
		bool setBDim(vector<unsigned int> const &dim);

		/**
		 * \brief Set method for halls possible dimensions.
		 * \param dim New pair of intervals of possible dimensions.
		 * \return Returns true if the assignation was successfull, otherwise false.
		*/
		bool setHDim(vector<unsigned int> const &dim);

		/**
		 * \brief Set method for rooms possible dimensions.
		 * \param dim New pair of intervals of possible dimensions.
		 * \return Returns true if the assignation was successfull, otherwise false.
		*/
		bool setRDim(vector<unsigned int> const &dim);

		/**
		 * \brief Set method for subsections possible dimensions.
		 * \param dim New pair of intervals of possible dimensions.
		 * \return Returns true if the assignation was successfull, otherwise false.
		*/
		bool setSDim(vector<unsigned int> const &dim);

		/**
		 * \brief Method for creating an environment based on the configuration parameters of this class. With the configuration parameters, one might establish the range of possible values buildings' width & height might have. One can also determine how many buildings will be created, & if they should be connected by hall, a door, or let the method dertermine that undeterministically for each connection.
		 * \param json_file Name of the JSON file that will contain the hierarchical structure & spatial relation pairs of the generated environment.
		 * \param img_file Name of the output image file. If no argument is passed the no image is rendered.
		 * \return Returns true if both input parameters are valid file names, which means that the environment will successfully be created, otherwise false.
		*/
		bool generateEnv(string const &json_file, string const &img_file = "non");

		vector<std::string> splitStr(std::string const &fullString,std::string const &delimiter);
};

#endif
