#include <unordered_map>
#include <algorithm>
#include <vector>
#include <list>
#include <random>
#include <chrono>
#include <ctime>
#include <iostream>
#include <string>
#include <assert.h>

using namespace std;

template <class T>
inline void hash_combine(std::size_t & seed, const T & v)
{
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std
{
template<typename S, typename T> struct hash<pair<S, T>>
{
	inline size_t operator()(const pair<S, T> & v) const
	{
		size_t seed = 0;
		::hash_combine(seed, v.first);
		::hash_combine(seed, v.second);
		return seed;
	}
};
}

int main(int argc, char** argv)
{

	// initialize datastructures
	unordered_map<pair<int,int>, int> myMap; // @suppress("Invalid template argument")

	myMap[make_pair(1,1)] = 10;
	myMap[make_pair(1,2)] = 2;
	myMap[make_pair(2,5)] = 3;
	myMap[make_pair(3,10)] = 22;


	for(unordered_map<pair<int,int>, int>::iterator it = myMap.begin(); it!=myMap.end(); it++) {

		cout << "key1: " << it->first.first << "; key2: " << it->first.second << "; value: " << myMap[it->first] << endl;
	}


	return 0;
}
