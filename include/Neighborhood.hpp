/**
 * \class Neighborhood
 * 
 * \brief This class is for representing neighborhood relation pairs between states.
 * 
 * \author $Author: Sergio A. Serrano$
 * 
 * \date $Date: 21/04/19$
 * 
 * Contact: sserrano@inaoep.mx
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <exception>
#include <nlohmann/json.hpp>
#include <TreeHandle.hpp>

using namespace std;

#ifndef NEIGHBORHOOD
#define NEIGHBORHOOD

class Neighborhood
{
	private:

		vector<string> states;
		vector< vector<int> > neig_to;

		//For the specific case of navigation problems
		vector< vector<int> > to_left_of_;
		vector< vector<int> > to_right_of_;
		vector< vector<int> > above_;
		vector< vector<int> > below_;

		/**
		 * \brief This method is for removing the (") characters from a string.
		 * \param s Original string to be cleaned.
		 * \return Returns a version of 's' without the quotes, therefore, for each quote in 's' the output string will be one character shorter.
		*/
		string stripQuotes(string const &s);

	public:

		/**
		 * \brief Default constructor.
		*/
		Neighborhood();

		/**
		 * \brief This method is for loading pairs of states for the 'above' and 'to-left-of' relations in the navigation problem.
		 * \param file_json Path to the JSON file from which the relation pairs shall be loaded.
		 * \return Returns true the loading process succeeded, otherwise false.
		*/
		bool navFromJson(string file_json);

		/**
		 * \brief Not implemented yet.
		 * \param file_name
		 * \return 
		*/
		bool save(string file_name);

		/**
		 * \brief Not implemented yet.
		 * \param file_name
		 * \return 
		*/
		bool load(string file_name);

		/**
		 * \brief This method is for getting all the states that are neighbors of state 's'.
		 * \param s State whose neighborhood is being requested.
		 * \param success Reference to a variable to determine if s' neighbors were returned.
		 * \return Returns a vector of strings containing the states in s' neighborhood.
		*/
		vector<string> neigTo(string const &s, bool &success);

		/**
		 * \brief This method returns the set of states (if any) located at the left of 's'.
		 * \param s State whose left-neighbor is being consulted.
		 * \param success Reference to a flag that will notice if such neighbor was found.
		 * \return Returns the neigborhood of states that are at the left of 's'.
		*/
		vector<string> leftOf(string const &s, bool &success);

		/**
		 * \brief This method returns the set of states (if any) located at the right of 's'.
		 * \param s State whose right-neighbor is being consulted.
		 * \param success Reference to a flag that will notice if such neighbor was found.
		 * \return Returns the neigborhood of states that are at the right of 's'.
		*/
		vector<string> rightOf(string const &s, bool &success);

		/**
		 * \brief This method returns the set of states (if any) located above 's'.
		 * \param s State whose above-neighbor is being consulted.
		 * \param success Reference to a flag that will notice if such neighbor was found.
		 * \return Returns the neigborhood of states that are above 's'.
		*/
		vector<string> aboveOf(string const &s, bool &success);

		/**
		 * \brief This method returns the set of states (if any) located below 's'.
		 * \param s State whose below-neighbor is being consulted.
		 * \param success Reference to a flag that will notice if such neighbor was found.
		 * \return Returns the neigborhood of states that are below 's'.
		*/
		vector<string> belowOf(string const &s, bool &success);

		/**
		 * \brief This method returns the set of states that make up its neighborhood, but it excludes those states conatined in the 'exc' vector parameter.
		 * \param s State whose neighborhood is being consulted.
		 * \param exc Set of states that will be filtered out (if they appear) from s' neighborhood.
		 * \return Returns the filtered neighborhood of s.
		*/
		vector<string> neigToExc(string const &s, vector<string> const &exc);

		/**
		 * \brief This method propagates the 'to-left-of', 'to-right-of', 'above' and 'below' relations from the states at the bottom of the hierarchical structure all the way up to the states the are in the level immediately below the rot node.
		 * \param th A reference to the TreeHandle object that contains the hierarchical description of state space.
		 * \return Returns the tree has at least two levels besides the one at which root is located, otherwise false.
		*/
		bool propNeigNav(TreeHandle &th);

		/**
		 * \brief This recursive method computes the shortest path to 'target'. When it is first invoked to start the search, path must contain the state from which the search will start, also 'lbound' should not be modified, this parameter is required for recursive callings. Moreover, due to its computational cost, this methoed should only be used to find a path between building states.
		 * \param path Reference to a vector that will hold a path to the target state if found.
		 * \param target The target state trying to be reached.
		 * \param lbound Lower bounder parameter that holds the length of the shortest path that reached the target state so far. This bound helps to avoid exploring innecessary larger paths.
		 * \return Returns true if a 'path' contains a path to the target state.
		*/
		bool recPath(vector<string> &path, string const &target, int const &lbound = -1);

		/**
		 * \brief This method is an overload that enables to bound the search within a set of states defined by 'subspace', that is, this method does not explores paths with states that are  not in 'subspace'.
		 * \param subspace Set of states that bounds  the path search space.
		 * \return Returns true if a 'path' contains a path to the target state.
		*/
		bool recPath(vector<string> const &subspace, vector<string> &path, string const &target, int const &lbound = -1);
};

#endif
