#include "atomic_file.hpp"

#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

namespace gamescope::framegen
{

AtomicOutputFile::AtomicOutputFile( std::string finalPath )
	: m_finalPath( std::move( finalPath ) )
{
	if ( m_finalPath.empty() )
	{
		m_error = EINVAL;
		return;
	}

	m_tempPath = m_finalPath;
	m_tempPath += ".tmp.XXXXXX";
	const int fd = mkostemp( m_tempPath.data(), O_CLOEXEC );
	if ( fd < 0 )
	{
		m_error = errno;
		m_tempPath.clear();
		return;
	}

	m_file = fdopen( fd, "wb" );
	if ( m_file == nullptr )
	{
		const int openError = errno;
		close( fd );
		unlink( m_tempPath.c_str() );
		m_tempPath.clear();
		m_error = openError;
	}
}

AtomicOutputFile::~AtomicOutputFile()
{
	close_and_discard();
}

void AtomicOutputFile::remember_error( int error )
{
	if ( m_error == 0 )
		m_error = error != 0 ? error : EIO;
}

void AtomicOutputFile::close_and_discard()
{
	if ( m_file != nullptr )
	{
		if ( fclose( m_file ) != 0 )
			remember_error( errno );
		m_file = nullptr;
	}
	if ( !m_tempPath.empty() )
	{
		unlink( m_tempPath.c_str() );
		m_tempPath.clear();
	}
}

bool AtomicOutputFile::write( const void *data, size_t size )
{
	if ( m_file == nullptr || m_error != 0 )
		return false;
	if ( size == 0 )
		return true;
	if ( data == nullptr )
	{
		remember_error( EINVAL );
		return false;
	}

	errno = 0;
	if ( fwrite( data, 1, size, m_file ) != size )
	{
		remember_error( errno );
		return false;
	}
	return true;
}

bool AtomicOutputFile::commit()
{
	if ( m_file == nullptr )
	{
		remember_error( EBADF );
		return false;
	}
	if ( m_error != 0 )
	{
		close_and_discard();
		return false;
	}

	errno = 0;
	if ( fclose( m_file ) != 0 )
		remember_error( errno );
	m_file = nullptr;
	if ( m_error != 0 )
	{
		close_and_discard();
		return false;
	}

	if ( rename( m_tempPath.c_str(), m_finalPath.c_str() ) != 0 )
	{
		remember_error( errno );
		close_and_discard();
		return false;
	}
	m_tempPath.clear();
	return true;
}

} // namespace gamescope::framegen
