/* compile with:
 *
 *      gcc -O2 -Wall pyr.c `pkg-config vips --cflags --libs` -o pyr
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
	VipsImage *blurred, *scaled;
	int filter_radius = 4, scale, i, nlevels=0;
	char fname[255];

	if( vips_init( argv[0] ) )
		vips_error_exit( NULL ); 

	context = g_option_context_new( "pyr infile basename nlevels - Gaussian pyramid" );

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

	if( argc != 4 )
		vips_error_exit( "usage: %s infile basename nlevels", argv[0] ); 
	
	if( !(in = vips_image_new_from_file( argv[1] )) )
		vips_error_exit( NULL );

	nlevels = atoi(argv[3]);

	printf( "image size:  width = %d\theight = %d\n", 
		vips_image_get_width( in ), vips_image_get_height(in) ); 

	scale = 1;
	for (i=0; i < nlevels; ++i) {
		if ( vips_gaussblur( in, &blurred, (int)floor(filter_radius*pow(sqrt(2), i)), NULL ) )
			vips_error_exit( NULL );

		scale *= 2;
		if (vips_shrink(blurred, &scaled, scale, scale, NULL))
			vips_error_exit( NULL );
		g_object_unref(blurred);		
		sprintf(fname, "%s-level_%d.ppm", argv[2], i+1);

		if( vips_image_write_to_file(scaled, fname))
			vips_error_exit( NULL );
		g_object_unref(scaled); 
	}

	g_object_unref( in ); 
        
	return( 0 );
}

