#include <iostream>
#include <string>
#include <vector>
#include <TreeHandle.hpp>
#include <HPomdp.hpp>

int main(int argc, char** argv)
{
	HPomdp hp;
	if(argc == 2) hp.renderExecutionHistory(string(argv[1]));
	else hp.run_eir(argc, argv);

	return 0;
}
