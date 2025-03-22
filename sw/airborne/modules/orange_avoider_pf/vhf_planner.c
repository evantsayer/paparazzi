#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <math.h>

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
    double potential;  // Potential function for obstacle avoidance
} GridHistogramCell;

typedef struct {
    int polar_obstacle_density[N_SECTORS];  // Depth intensity histogram
    double potential;  // Potential function for obstacle avoidance
} PolarHistogram;

typedef struct {
    int polar_obstacle_density[N_SECTORS];  // Depth intensity histogram
    double potential;  // Potential function for obstacle avoidance
} PolarHistogram;

// Mapping function that maps the grid histogram to the polar historam
void update_grid_histogram(GridHistogramCell grid[N_X_GRIDS][N_Y_GRIDS], Stripe image[N_STRIPES]) {
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
void map_grid_to_polar_histogram(GridHistogramCell grid[N_X_GRIDS][N_Y_GRIDS], PolarHistogram polar) {
    float x_d;
    float y_d;
    int sector_index;
    GridHistogramCell cell;
    // Set obstacle density values to zero
    memset(polar.polar_obstacle_density, 0, sizeof(polar.polar_obstacle_density));

    for (int i=0; i<N_X_GRIDS; i++) {
        for (int j=0; i<N_Y_GRIDS; j++) {
            cell = grid[i][j];

            float beta = (cell.center.y-y_d)/(cell.center.x-x_d);
            float d = (cell.center.y-y_d)*(cell.center.y-y_d) + (cell.center.x-x_d)*(cell.center.x-x_d);
            float m = cell.confidance*cell.confidance*(a - b*d);
            sector_index = SECTOR_WIDTH*(int)(beta/SECTOR_WIDTH);
            polar.polar_obstacle_density[sector_index] += m; 
        }
    }
}

void smooth_polar_histogram(PolarHistogram poler, int smoothing_factor) {
    float smoothened_density;
    for (int i=0; i<N_SECTORS; i++) {
        smoothened_density = 0;
        for (int j=0; j<smoothing_factor; j++) {
            smoothened_density += ;
        }
    }
}

// Smooth the polar histogram using a moving average filter
void smooth_histogram() {
    PolarCell smoothed[ANGLE_BINS];
    memcpy(smoothed, polar_grid, sizeof(polar_grid));
    
    for (int i = SMOOTHING_WINDOW; i < ANGLE_BINS - SMOOTHING_WINDOW; i++) {
        double sum = 0;
        for (int j = -SMOOTHING_WINDOW; j <= SMOOTHING_WINDOW; j++) {
            sum += polar_grid[i + j].potential;
        }
        smoothed[i].potential = sum / (2 * SMOOTHING_WINDOW + 1);
    }
    
    memcpy(polar_grid, smoothed, sizeof(polar_grid));
}

// Identify candidate directions where obstacle density is low
void get_all_candidate_directions(PolarHistogram *polar, int *candidates[MAX_CANDIDATE_DIRECTIONS]) {
    int num_candidates = 0;
    for (int i = 0; i < N_SECTORS; i++) {
        if (polar[i].potential < OBSTACLE_THRESHOLD) {
            candidates[num_candidates++] = i;
        }
        if (num_candidates > MAX_CANDIDATE_DIRECTIONS) {
            break;
        }
    }
}

// Find the best direction to move based on the goal alignment
int find_best_direction(Point* position, Point* target, PolarHistogram* polar) {
    float target_angle = 
    int candidates[ANGLE_BINS];
    int num_candidates;
    find_candidate_directions(candidates, &num_candidates);
    
    int best_angle = GOAL_ANGLE;
    double min_cost = 1e9;
    
    for (int i = 0; i < num_candidates; i++) {
        int angle = candidates[i];
        double cost = fabs(angle - GOAL_ANGLE) + polar_grid[angle].potential;
        if (cost < min_cost) {
            min_cost = cost;
            best_angle = angle;
        }
    }
    return best_angle;
}

void scan_field() {

}

// Main function to simulate navigation
int init() {
    scan_field();
}

int main() {
    Point position;
    Point target;
    update_grid_histogram();
    PolarHistogram polar = map_grid_to_polar_histogram();
    int direction = find_best_direction(&position, &target, &polar);
    
}
