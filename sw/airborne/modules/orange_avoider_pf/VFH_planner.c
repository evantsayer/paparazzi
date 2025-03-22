#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <math.h>
#include "firmwares/rotorcraft/navigation.h"

#define N_X_GRIDS 50 
#define N_Y_GRIDS 50
#define N_SECTORS 20
#define SECTOR_WIDTH 2 // Angular width of the sectors
#define a 1.
#define b 1.
#define N_STRIPES 20
#define CELL_X_WIDTH 2 // x width of a cell in the cortesian grid
#define CELL_Y_WIDTH 2 // y width of a cell in the cortesian grid
#define OBSTACLE_THRESHOLD 2 //
#define MAX_CANDIDATE_DIRECTIONS 10

enum PlanningState {
    SAFE,
    OBSTACLE_FOUND,
    SEARCH_FOR_SAFE_HEADING,
    OUT_OF_BOUNDS
}

typedef struct {
    float x;
    float y;
} Point;

typedef struct {
    float alpha_min;
    float alpha_max;
    float depth;
} Stripe;

typedef struct {
    int confidance;  // Depth intensity histogram
    Point center;
} GridHistogramCell;

typedef struct {
    int polar_obstacle_density[N_SECTORS];  // Depth intensity histogram
} PolarHistogram;

typedef struct {
    int polar_obstacle_density[N_SECTORS];  // Depth intensity histogram
} PolarHistogram;

// Define global variables
GridHistogramCell grid[N_X_GRIDS][N_Y_GRIDS];
memset(grid, 0, sizeof(grid));
PolarHistogram polar;


// Mapping function that maps the grid histogram to the polar historam
void update_grid_histogram(GridHistogramCell *grid[N_X_GRIDS][N_Y_GRIDS], Stripe *image[N_STRIPES]) {
    float alpha_min;
    float alpha_max;
    float depth;
    Point p0;
    Point p1;
    
    for (int i =0; i<N_STRIPES; i++) {
        alpha_min = image[i].alpha_min;
        alpha_max = image[i].alpha_max;
        depth = image[i].depth;

        p0.x = depth*cos(alpha_min);
        p0.y = depth*sin(alpha_min);
        p1.x = depth*cos(alpha_max);
        p1.y = depth*sin(alpha_max);

        // Apply Braham's algorithm to draw line between two sector separator points
        //TODO: test this algo, expecially for negative values
        int discrete_x0 = (int)(p0.x/CELL_X_WIDTH);
        int discrete_y0 = (int)(p0.y/CELL_Y_WIDTH);
        int discrete_x1 = (int)(p1.x/CELL_X_WIDTH);
        int discrete_y1 = (int)(p1.y/CELL_Y_WIDTH);

        int dx = discrete_x1 - discrete_x0;
        int dy = discrete_y1 - discrete_y0;
        float D = 2*dy - dx;
        int y = discrete_y0;

        int x_steps = (int)(dx/CELL_X_WIDTH);
        int y_steps = (int)(dy/CELL_Y_WIDTH);

        for (int j=discrete_x0; j<discrete_x0+x_steps; j++) {
            grid[discrete_x0+j][y].confidance += ;
            if (D > 0) {
                y += 1;
                D -= 2*dx;
            }
            D += 2*dy;
        }
    }
}

// Mapping function that maps the polar histogram to the grid historam
void map_grid_to_polar_histogram(GridHistogramCell *grid[N_X_GRIDS][N_Y_GRIDS], PolarHistogram *polar) {
    float x_d;
    float y_d;
    int sector_index;
    GridHistogramCell *cell;
    // Set obstacle density values to zero
    memset(polar.polar_obstacle_density, 0, sizeof(polar.polar_obstacle_density));

    for (int i=0; i<N_X_GRIDS; i++) {
        for (int j=0; i<N_Y_GRIDS; j++) {
            cell = grid[i][j];

            float beta = atan2(cell.center.y-y_d)/(cell.center.x-x_d);
            float d = (cell.center.y-y_d)*(cell.center.y-y_d) + (cell.center.x-x_d)*(cell.center.x-x_d);
            float m = cell.confidance*cell.confidance*(a - b*d);
            sector_index = SECTOR_WIDTH*(int)(beta/SECTOR_WIDTH);
            polar.polar_obstacle_density[sector_index] += m; 
        }
    }
}

// Smoothing function for the polar histogram
void smooth_polar_histogram(PolarHistogram *polar, int smoothing_radius) {
    /**
     * Function smoothing the polar histogram
     * 
     */
    float smoothened_density;

    for (int sector = 0; sector < N_SECTORS; sector++) {
        smoothened_density = 0;
        // Take the weighted sum of obstacle density values about the current sector in a radius of smoothing_radius
        for (int i = sector; i < sector + smoothing_radius; i++) {
            smoothened_density += (smoothing_radius+sector-i)*polar->polar_obstacle_density[i]
        }
        for (int i = sector; i < sector + smoothing_radius+1; --i) {
            smoothened_density += (smoothing_radius+sector-i)*polar->polar_obstacle_density[i]
        }
        polar->polar_obstacle_density = smoothened_density/(2*smoothing_radius + 1)
    }
}

// Identify candidate directions where obstacle density is low
void get_all_candidate_directions(PolarHistogram *polar, int *candidates[MAX_CANDIDATE_DIRECTIONS], int *num_candidates) {
    for (int i = 0; i < N_SECTORS; i++) {
        if (polar[i].potential < OBSTACLE_THRESHOLD) {
            candidates[*num_candidates++] = i;
        }
        if (*num_candidates > MAX_CANDIDATE_DIRECTIONS) {
            break;
        }
    }
}

// Find the best direction to move based on the target point
float find_best_direction(Point* position, Point* target, PolarHistogram* polar) {

    float target_angle = acos((target->x * position->x + target->y * position->y)
        /(sqrt(pow(target->x-position->x, 2) + pow(target->y-position->y, 2))));

    int candidates[MAX_CANDIDATE_DIRECTIONS];
    int num_candidates = 0;
    find_candidate_directions(polar, candidates, num_candidates);
    
    double min_cost = 1e9;
    int angle;
    double cost;
    float best_angle;
    for (int i = 0; i < num_candidates; i++) {
        angle = candidates[i];
        cost = fabs(angle - target_angle) + polar[angle].potential;
        if (cost < min_cost) {
            min_cost = cost;
            best_angle = angle;
        }
    }
    return best_angle;
}

// Calculate max safe distance in direction
float get_max_safe_distance(Point* position, float direction) {

    Point p0;
    p0.x = depth*cos(alpha_min);
    p0.y = depth*sin(alpha_min);
    float max_distance;
    float current_safety = infinity;
    while (OBSTACLE_THRESHOLD < current_safety || max_distance < distance_to_safety_barrier ) {

    }
    alpha_min = image[i].alpha_min;
    alpha_max = image[i].alpha_max;
    depth = image[i].depth;

    
    p1.x = depth*cos(alpha_max);
    p1.y = depth*sin(alpha_max);

    // Apply Braham's algorithm to draw line between two sector separator points
    //TODO: test this algo, expecially for negative values
    int discrete_x0 = (int)(p0.x/CELL_X_WIDTH);
    int discrete_y0 = (int)(p0.y/CELL_Y_WIDTH);
    int discrete_x1 = (int)(p1.x/CELL_X_WIDTH);
    int discrete_y1 = (int)(p1.y/CELL_Y_WIDTH);

    int dx = discrete_x1 - discrete_x0;
    int dy = discrete_y1 - discrete_y0;
    float D = 2*dy - dx;
    int y = discrete_y0;

    int x_steps = (int)(dx/CELL_X_WIDTH);
    int y_steps = (int)(dy/CELL_Y_WIDTH);

    for (int j=discrete_x0; j<discrete_x0+x_steps; j++) {
        grid[discrete_x0+j][y].confidance += ;
        if (D > 0) {
            y += 1;
            D -= 2*dx;
        }
        D += 2*dy;
    }
}

Point get_multiple_waypoints() {}

void scan_field() {
    return;
}

void get_full_state(Point *position, Pose *pose) {

}

// Main function to simulate navigation
int init() {
    scan_field();
}

int main() {
    Point position;
    Point target;
    update_grid_histogram();
    PolarHistogram polar;
    map_grid_to_polar_histogram(grid, polar, );
    
    float direction = find_best_direction(&position, &target, &polar);
    
}
