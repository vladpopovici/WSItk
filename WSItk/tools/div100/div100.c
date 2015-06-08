/* compile with:
 *
 *      gcc -O3 -Wall div100.c `pkg-config vips --cflags --libs` -o div100
 */

#include <stdio.h>
#include <stdlib.h>
#include <vips/vips.h>
#include <math.h>

int
main( int argc, char **argv )
{
	GOptionContext *context;
	GOptionGroup *main_group;
	GError *error = NULL;
	VipsImage *in;
	VipsImage *out;

	if( vips_init( argv[0] ) )
		vips_error_exit( NULL ); 

	context = g_option_context_new( "div100 infile outfile - Resize an image to 100x smaller." );

	main_group = g_option_group_new( NULL, NULL, NULL, NULL, NULL );
	g_option_context_set_main_group( context, main_group );
	g_option_context_add_group( context, vips_get_option_group() );

	if( !g_option_context_parse( context, &argc, &argv, &error ) ) {
		if( error ) {
			fprintf( stderr, "%s\n", error->message );
			g_error_free( error );
		}

		vips_error_exit( NULL );
	}

	if( argc != 3 )
		vips_error_exit( "usage: %s infile outfile", argv[0] ); 
	
	if( !(in = vips_image_new_from_file( argv[1] )) )
		vips_error_exit( NULL );

	if (vips_shrink(in, &out, 100, 100, NULL))
		vips_error_exit( NULL );

       	g_object_unref( in ); 

	if( vips_image_write_to_file( out, argv[2] ) )
		vips_error_exit( NULL );

	g_object_unref( out ); 	

	return( 0 );
}

