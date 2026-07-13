#pragma once

#include <cstddef>
#include <cstdio>
#include <string>

namespace gamescope::framegen
{

// Buffered output committed with one same-directory rename. The unique staging
// name prevents concurrent Gamescope instances from sharing a partial file;
// abandoning the object leaves the previous destination untouched.
class AtomicOutputFile
{
public:
	explicit AtomicOutputFile( std::string finalPath );
	~AtomicOutputFile();

	AtomicOutputFile( const AtomicOutputFile & ) = delete;
	AtomicOutputFile &operator=( const AtomicOutputFile & ) = delete;
	AtomicOutputFile( AtomicOutputFile && ) = delete;
	AtomicOutputFile &operator=( AtomicOutputFile && ) = delete;

	[[nodiscard]] bool is_open() const { return m_file != nullptr; }
	[[nodiscard]] int error() const { return m_error; }

	bool write( const void *data, size_t size );
	bool commit();

private:
	void remember_error( int error );
	void close_and_discard();

	std::string m_finalPath;
	std::string m_tempPath;
	FILE *m_file = nullptr;
	int m_error = 0;
};

} // namespace gamescope::framegen
