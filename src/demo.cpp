#include <HPomdp.hpp>

int main(int argc, char** argv)
{
	//Run the flat POMDP, two-level POMDP and hierarchical architecture
	HPomdp hp;
	hp.run(argc,argv);

	return 0;
}
