/**
 * \class TreeHandle
 * 
 * \brief This class is for performing actions on a hierarchical tree structure, such as inserting and reading elements.
 * 
 * \author $Author: Sergio A. Serrano$
 * 
 * \date $Date: 20/04/19$
 * 
 * Contact: sserrano@inaoep.mx
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <exception>
#include <cmath>
#include <st_tree.h>
#include <nlohmann/json.hpp>

using namespace st_tree;
using namespace std;

#ifndef TREE_HANDLE
#define TREE_HANDLE

class Neighborhood;
class TreeHandle
{
	private:

		tree<unsigned, keyed<string>> t_;

		//Vectors of building & hall dimensions used for computing
		//the shortest distance between 2 cells
		vector<string> b_name;
		vector<vector<unsigned> > b_dim;
		vector<string> h_name;
		vector<vector<unsigned> > h_dim;

		/**
		 * \brief This method is for removing the (") characters from a string.
		 * \param s Original string to be cleaned.
		 * \return Returns a version of 's' without the quotes, therefore, for each quote in 's' the output string will be one character shorter.
		*/
		string stripQuotes(string const &s);

		/**
		 * \brief This method for recursively traverse the tree in a depth-first way, to write the node's key in a text file. It first writes the key of the node to which 'ite' is pointing to, then it writes the keys of its children nodes, by calling this same method.
		 * \param outfile Stream to the already open file where data is being written.
		 * \param d Depth of node to which 'ite' is pointing to.
		 * \param ite Iterator pointing to node whose key has not been written.
		*/
		void recSave(ofstream &outfile, unsigned d, tree<unsigned, keyed<string> >::node_type::iterator &ite);

		/**
		 * \brief This method is for recursively build a tree structure by reading data from a text file.
		 * \param n A reference to the last node inserted into the tree structure.
		 * \param vec Vector that contains all the lines read from the text file.
		 * \param index Index of the next line (from vector 'vec') to be inserted as a node in the tree.
		 * \return Returns true if all the calls to this method, including itself, have inserted nodes successfully, otherwise returns false.
		*/
		bool recLoad(tree<unsigned, keyed<string> >::node_type &n, vector<string> const &vec, unsigned int &index);

	public:

		/**
		 * \brief Default and only constructor to the class, which initializes the tree structure by inserting the value to its root node.
		*/
		TreeHandle();

		/**
		 * \brief This method inserts a node as a child of the root node.
		 * \param node_key Key value of the node to be inserted, that is, its unique identifier.
		 * \param val Data value of the node to be inserted.
		*/
		void insert(string node_key, unsigned val);

		/**
		 * \brief This method inserts a node as a child of the node that has 'parent_key' as key value, if any.
		 * \param parent_key Key value of the node that must exist in the tree in order to insert the new node as its child.
		 * \param node_key Key value of the node to be inserted, that is, its unique identifier.
		 * \param val Data value of the node to be inserted.
		 * \return Returns true if the parent node was found, otherwise false.
		*/
		bool insert(string parent_key, string node_key, unsigned val);

		/**
		 * \brief This method is for consulting the data value of a given node.
		 * \param node_key Key value of the node whose data is being queried.
		 * \param output_val Reference to the variable that will hold the value of the queried data.
		 * \return Returns true the requested node exists in the tree, otherwise false.
		*/
		bool query(string node_key, unsigned &output_val);

		/**
		 * \brief This method is for savin the current tree structure in text file.
		 * \param file_name Path to the file in which the tree shall be saved.
		 * \return Returns true if the saving process succeeded, otherwise false.
		*/
		bool save(string file_name);

		/**
		 * \brief This method is for loading a tree structure from at text file (that was generated with the save method).
		 * \param file_name Path to the file from which the tree shall be loaded.
		 * \return Returns true the loading process succeeded, otherwise false.
		*/
		bool load(string file_name);

		/**
		 * \brief This method is for loading a tree structure from a JSON file for the specific task of navigation.
		 * \param file_json Path to the JSON file from which the tree shall be loaded.
		 * \return Returns true the loading process succeeded, otherwise false.
		*/
		bool navFromJson(string file_json);

		/**
		 * \brief This method is for consulting the current tree's max depth.
		 * \return Returns the amount of levels the tree has.
		*/
		unsigned int depth() const;

		/**
		 * \brief This method is for geting all the nodes' key values that are located at a given depth (or level).
		 * \param level Depth from which nodes will be gathered.
		 * \return Returns a vector of strings containing the keys of those nodes located at the requested level.
		*/
		vector<string> keysAtLevel(unsigned int level);

		/**
		 * \brief This method is for geting all the nodes' key values that are children of a given node.
		 * \param parent_key Key value of the node whose children will be gathered.
		 * \return Returns a vector of strings containing the keys of those nodes that are children of 'parent_key'.
		*/
		vector<string> keysOfChildren(string parent_key);

		/**
		 * \brief This method is for consulting if a node is children of another one.
		 * \param parent_key Key value of the node to be evaluated as the parent.
		 * \param child_key Key value of the node to be evaluated as the child.
		 * \return Returns true if the node conatining 'parent_key' is parent of the node containing 'child_key'.
		*/
		bool isParent(string parent_key, string child_key);

		/**
		 * \brief This method is determining if there is a downwards path that starts in 'ancestor_key' node and leads to 'child_key' node, that is, if the first one is an ancestor of the second one.
		 * \param ancestor_key Key value of the node to be evaluated as the ancestor.
		 * \param child_key Key value of the node to be evaluated as the descendant.
		 * \return Returns true if the node conatining 'ancestor_key' is ancestor of the node containing 'child_key'.
		*/
		bool isAncestor(string ancestor_key, string child_key);

		/**
		 * \brief This method is for consulting if a node exists in the tree.
		 * \param node_key Key value of the node to be searched.
		 * \return Returns true if the node is within the tree, otherwise false.
		*/
		bool inTree(string const &node_key);

		/**
		 * \brief This method is for consulting the level at which a node is in the tree.
		 * \param node_key Key value of the node to be evaluated.
		 * \return Returns -1 if the node does not exist in the tree, otherwise the level at which it is located.
		*/
		int ply(string const &node_key);

		/**
		 * \brief This method is for building a path from the root node to the node that has for key value 'node_key'.
		 * \param node_key Key value of the node set as the last node in the path.
		 * \return Returns a vector of strings representing the key values of those nodes that are part of the path that starts at root and ends in 'node_key'.
		*/
		vector<string> hieState(string const &node_key);

		/**
		 * \brief This method is for determining the deepest common ancestor of node A and node B.
		 * \param node_a Key value of node A.
		 * \param node_b Key value of node B.
		 * \return Returns a tuple containing the key and ply values of the deepest common ancestor between node A and B.
		*/
		tuple<string, int> dca(string const &node_a, string const &node_b);

		/**
		 * \brief This method is for consulting the parent node of the one that has for key value 'node_key'.
		 * \param node_key Key value of the child node.
		 * \param success Reference to a variable that will hold the result of finding the parent node in the tree.
		 * \return Returns the key value of the parent node if found, otherwise an empty string.
		*/
		string parent(string const &node_key, bool &success);

		/**
		 * \brief This method computes the Manhattan distance between the bordering cells within 'room'. The pair of borders should be different & specified by one of the following characters: l,r,u,d.
		 * \param nh Reference to a Neighborhood object holding the connectivity pairs of the environment stored in 'this' hierarchy.
		 * \param bor1 A border of the room to be analyzed <l,r,u,d>.
		 * \param bor2 A border of the room to be analyzed <l,r,u,d>.
		 * \param room Name of the room to be analyzed.
		 * \return Returns a vector of pairs of cells that belong to 'bor1' & 'bor2', as well as the Manhattan distance between them. 
		*/
		vector<tuple<string,string,int> > borPaths(Neighborhood &nh, string const &bor1, string const &bor2, string const &room);

		/**
		 * \brief This method computes the Manhattan distance between a cell and the bordering cells of a wall within 'room'. The border should be specified by one of the following characters: l,r,u,d.
		 * \param nh Reference to a Neighborhood object holding the connectivity pairs of the environment stored in 'this' hierarchy.
		 * \param init_cell A cell within 'room' to measure its Manhattan distance to every border cell in the wall indicated by 'bor'.
		 * \param bor The border of the room to be analyzed <l,r,u,d>.
		 * \param room Name of the room to be analyzed.
		 * \return Returns a vector of pairs of cells that belong to 'init_cell' & 'bor', as well as the Manhattan distance between them. 
		*/
		vector<tuple<string,string,int> > cellBorPaths(Neighborhood &nh, string const &init_cell, string const &bor, string const &room);

		/**
		 * \brief This method is for computing the manhattan distance between two cells (A and B) that are in the same room.
		 * \param nh Reference to a Neighborhood object holding the connectivity pairs of the environment stored in 'this' hierarchy.
		 * \param sa Cell A.
		 * \param sb Cell B.
		 * \return Returns the Manhattan distance between cells A and B.
		*/
		int manhDist(Neighborhood &nh, string const &sa,string const &sb);

		/**
		 * \brief This method recursively computes the length of the shortest path between two cells that are in the same building. When first invoked, 'path' should contain a single tuple holding the following values: room that contains the initial cell, initial cell, empty-string and -1. Also, one should not set the 'ubound' since this parameter is required by the recursive calls.
		 * \param nh Reference to a Neighborhood object holding the connectivity pairs of the environment stored in 'this' hierarchy.
		 * \param path Reference to a vector that will hold the shortest path between the initial cell and the target cell. Each element in 'path' has the following values: room that contains the cell-pair represented by this tuple, initial cell of cell-pair, target cell of cell-pair, Manhattan distance for this cell-pair.
		 * \param tgt_hc Hierarchical state of the target cell.
		 * \param ubound Parameter used by the recursive calls in order to bound th search of the shortest path.
		 * \return Returns true if a path that connects the initial and target cells was found  within their building, otherwise false.
		*/
		bool recRoomPath(Neighborhood &nh, vector<tuple<string,string,string,int> > &path, vector<string> const &tgt_hc, int const &ubound = -1);

		/**
		 * \brief This method is for computing the length of the shortest path between 'init_cell' and a cell within 'target_room' (in case 'target_sc' is a cell) or all the bordering cells of one of the walls of 'target_room' (in case 'target_sc' is a side).
		 * \param nh Reference to a Neighborhood object holding the connectivity pairs of the environment stored in 'this' hierarchy.
		 * \param init_cell Path's initial cell.
		 * \param target_room Room to be reached.
		 * \param target_sc Path's final cell within 'target_room' or one of the sides of  'target_room' that must be specified by one of the following charecters: l,r,u,d.
		 * \return Returns the length of the shortest path between 'init_cell' and 'target_sc' (in case it is a cell), or each of its bordering cells (in case it is a side of 'target_room').
		*/
		vector<vector<tuple<string,string,int,string> > > roomPath(Neighborhood &nh,string const &init_cell,string const &target_room,string const &target_sc);

		/**
		 * \brief This method is for computing the length of the shortest path between two cells (A and B).
		 * \param nh Reference to a Neighborhood object holding the connectivity pairs of the environment stored in 'this' hierarchy.
		 * \param sa Cell A.
		 * \param sb Cell B.
		 * \param display If set, the sequence of cell-pairs that constitute the shortest path will be displayed.
		 * \return Returns the length of the shortest path between cells A and B.
		*/
		int optPath(Neighborhood &nh, string const &sa, string const &sb, bool const &display = false);
};

#endif
