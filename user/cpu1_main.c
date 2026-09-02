/*********************************************************************************************************************
* TC264 Open Source Library
* Copyright (c) 2022 SEEKFREE Technology
*
* RT1064 image-recognition logic ported to TC264 CPU1.
* The SEEKFREE library remains licensed under GPL-3.0-or-later.
********************************************************************************************************************/

#include "zf_common_headfile.h"
#include <stdlib.h>

/*
 * Keep the rest of the proven project on its original Debug settings, but
 * compile this image-processing translation unit for speed.  Source-local
 * pragmas make the setting independent of which IDE build configuration is
 * selected and do not alter CPU0, ISR, PID, motor or safety code generation.
 */
#ifndef CPU1_PROCESSING_OPTIMIZE
#define CPU1_PROCESSING_OPTIMIZE       (1)
#endif

#if CPU1_PROCESSING_OPTIMIZE && defined(__TASKING__)
#pragma optimize 2
#pragma tradeoff 0
#elif CPU1_PROCESSING_OPTIMIZE \
   && (defined(__HIGHTEC__) || defined(__GNUC__))
#pragma GCC optimize ("O2")
#endif

#pragma section all "cpu1_dsram"

#define IMG_ROW                 (120)
#define IMG_COL                 (188)
#define SCAN_STEP               (1)
#define SAMPLE_ROW_COUNT        (10)
#define HORIZONTAL_SCAN_BOTTOM_OFFSET (30U)
#define HORIZONTAL_SCAN_START_ROW_LIMIT \
    (IMG_ROW - 1U - HORIZONTAL_SCAN_BOTTOM_OFFSET)
#define EDGE_RATIO_THRESHOLD    (250)
#define MIN_VALID_TRACK_WIDTH   (8)
#define MIN_VALID_SAMPLE_ROWS   (3)
#define REFERENCE_BAND_TOLERANCE (3)
#define REFERENCE_MIN_DEPTH     (20)
#define REFERENCE_MIN_BAND_WIDTH (3)

#if HORIZONTAL_SCAN_BOTTOM_OFFSET >= IMG_ROW
#error "HORIZONTAL_SCAN_BOTTOM_OFFSET must be smaller than IMG_ROW"
#endif

extern uint8 img_buf[IMG_ROW * IMG_COL];
extern volatile uint8 frame_ready;
extern volatile uint8 result_ready;
#if LOW_COMPUTE_CPU1_INTERRUPT_WAKE
extern volatile uint8 cpu1_idle_enabled;
#endif
extern volatile uint8 line_valid_result;
extern volatile uint16 shared_tem_max;
extern volatile uint16 shared_tem_min;
extern volatile uint16 z_result;
extern volatile uint16 reference_col_result;
extern volatile uint16 reference_boundary_row_result;
extern volatile uint16 sample_start_row_result;
extern volatile uint8 reference_valid_result;
extern volatile uint8 left_valid_count_result;
extern volatile uint8 right_valid_count_result;
extern volatile uint8 left_max_lost_run_result;
extern volatile uint8 right_max_lost_run_result;
extern volatile uint16 left_edge_average_result;
extern volatile uint16 right_edge_average_result;

static uint8 is_gradient_edge (int16 first, int16 second)
{
    int32 delta = (int32)first - (int32)second;
    int32 magnitude = (delta < 0) ? -delta : delta;
    int32 numerator = ((int32)first + (int32)second) * 10;
    int32 denominator = magnitude + 1;

    /*
     * With the unsigned 8-bit image samples used by this function, the
     * original integer expression
     *
     *   ((first + second) * 10) / (abs(delta) + 1) < 400
     *
     * is exactly equivalent to the multiplication-only comparison below.
     * This removes a variable integer division from the image scan without
     * changing any edge decision.
     */
    return (uint8)(numerator
                 < (int32)EDGE_RATIO_THRESHOLD * denominator);
}

/*
 * Several neighbouring columns normally have the same longest white run.
 * Select the centre of the widest near-best band instead of the first (left-
 * most) tied column. This makes the reference position usable for judging the
 * direction in which the road really extends.
 */
static uint16 find_reference_column (const uint8 *boundary_row,
                                     uint8 *reference_valid)
{
    uint8 minimum_row = IMG_ROW - 1;
    uint16 best_start = 0;
    uint16 best_end = 0;
    uint16 best_width = 0;
    int current_start = -1;
    int col;

    for(col = 0; col < IMG_COL; col++)
    {
        if(boundary_row[col] < minimum_row)
        {
            minimum_row = boundary_row[col];
        }
    }

    for(col = 0; col <= IMG_COL; col++)
    {
        uint8 in_best_band = 0;

        if(col < IMG_COL
        && boundary_row[col] <= minimum_row + REFERENCE_BAND_TOLERANCE)
        {
            in_best_band = 1;
        }

        if(in_best_band)
        {
            if(current_start < 0)
            {
                current_start = col;
            }
        }
        else if(current_start >= 0)
        {
            uint16 current_end = (uint16)(col - 1);
            uint16 current_width = current_end - (uint16)current_start + 1U;

            if(current_width > best_width)
            {
                best_width = current_width;
                best_start = (uint16)current_start;
                best_end = current_end;
            }
            current_start = -1;
        }
    }

    *reference_valid = (uint8)(((IMG_ROW - 1U - minimum_row)
                                >= REFERENCE_MIN_DEPTH)
                            && (best_width >= REFERENCE_MIN_BAND_WIDTH));

    if(!(*reference_valid))
    {
        return IMG_COL / 2U;
    }
    return (best_start + best_end) / 2U;
}

/*
 * This is the RT1064 get_xian algorithm with bounds checks added:
 *  1. Scan every column bottom-up and record its white-run endpoint.
 *  2. Choose the centre of the longest-white-column band as the reference.
 *  3. On ten nearby rows, search both track edges from that reference.
 *  4. Return average(left + right). The centered target is therefore 188.
 */
static uint16 get_xian (uint8 *image,
                        uint16 white_threshold,
                        uint16 black_threshold,
                        uint8 *line_valid,
                        uint16 *reference_col_output,
                        uint16 *reference_boundary_row_output,
                        uint16 *sample_start_row_output,
                        uint8 *reference_valid_output,
                        uint8 *left_valid_count_output,
                        uint8 *right_valid_count_output,
                        uint8 *left_max_lost_run_output,
                        uint8 *right_max_lost_run_output,
                        uint16 *left_edge_average_output,
                        uint16 *right_edge_average_output)
{
    static uint8 boundary_row[IMG_COL];
    uint16 reference_col;
    uint16 reference_boundary_row;
    uint16 reference_mid_row;
    uint16 sample_start_row;
    uint32 centre_sum = 0;
    uint32 left_sum = 0;
    uint32 right_sum = 0;
    uint8 valid_rows = 0;
    uint8 left_valid_count = 0;
    uint8 right_valid_count = 0;
    uint8 left_lost_run = 0;
    uint8 right_lost_run = 0;
    uint8 left_max_lost_run = 0;
    uint8 right_max_lost_run = 0;
    int col;
    int row;
    int sample;

    memset(boundary_row, IMG_ROW - 1, sizeof(boundary_row));

    for(col = 0; col < IMG_COL; col += SCAN_STEP)
    {
        const uint8 *current_pixel =
            &image[(IMG_ROW - 1) * IMG_COL + col];
        int16 current = *current_pixel;

        for(row = IMG_ROW - 1; row > SCAN_STEP; row -= SCAN_STEP)
        {
            const uint8 *above_pixel =
                current_pixel - SCAN_STEP * IMG_COL;
            int16 above = *above_pixel;

            if(above <= white_threshold
            && (current < black_threshold
             || is_gradient_edge(current, above)))
            {
                boundary_row[col] = (uint8)row;
                break;
            }

            /* The old "above" sample is the next iteration's "current". */
            current = above;
            current_pixel = above_pixel;
        }
    }

    reference_col = find_reference_column(boundary_row,
                                           reference_valid_output);
    *reference_col_output = reference_col;

    /*
     * Keep the original midpoint of the longest white column whenever it is
     * above the configured limit.  If that midpoint would be too close to the
     * bottom, start at row 89 (119 - 30).  A short white column whose endpoint
     * is already below row 89 must start from that endpoint, never outside the
     * detected white run.
     */
    reference_boundary_row = boundary_row[reference_col];
    reference_mid_row = IMG_ROW / 2U + reference_boundary_row / 2U;
    sample_start_row = reference_mid_row;
    if(sample_start_row > HORIZONTAL_SCAN_START_ROW_LIMIT)
    {
        sample_start_row = HORIZONTAL_SCAN_START_ROW_LIMIT;
    }
    if(sample_start_row < reference_boundary_row)
    {
        sample_start_row = reference_boundary_row;
    }
    *reference_boundary_row_output = reference_boundary_row;
    *sample_start_row_output = sample_start_row;

    for(sample = 0; sample < SAMPLE_ROW_COUNT; sample++)
    {
        uint16 left_edge = 0;
        uint16 right_edge = IMG_COL - 1;
        uint8 left_detected = 0;
        uint8 right_detected = 0;
        uint8 left_white_seen = 0;
        uint8 right_white_seen = 0;

        row = sample_start_row + sample;
        if(row >= IMG_ROW)
        {
            row = IMG_ROW - 1;
        }

        for(col = (int)reference_col; col < IMG_COL; col += SCAN_STEP)
        {
            int16 current;
            int16 next;

            if(col >= IMG_COL - SCAN_STEP)
            {
                right_edge = IMG_COL - 1;
                image[row * IMG_COL + right_edge] = 0;
                break;
            }

            current = image[row * IMG_COL + col];
            next = image[row * IMG_COL + col + SCAN_STEP];
            if(current > white_threshold || next > white_threshold)
            {
                right_white_seen = 1;
            }
            if(next > white_threshold)
            {
                continue;
            }
            if(right_white_seen
            && (current < black_threshold || is_gradient_edge(current, next)))
            {
                right_edge = (uint16)col;
                right_detected = 1;
                image[row * IMG_COL + col] = 0;
                if(col + 1 < IMG_COL)
                {
                    image[row * IMG_COL + col + 1] = 0;
                }
                break;
            }
        }

        for(col = (int)reference_col; col >= 0; col -= SCAN_STEP)
        {
            int16 current;
            int16 previous;

            if(col <= 0)
            {
                left_edge = 0;
                image[row * IMG_COL] = 0;
                break;
            }

            current = image[row * IMG_COL + col];
            previous = image[row * IMG_COL + col - SCAN_STEP];
            if(current > white_threshold || previous > white_threshold)
            {
                left_white_seen = 1;
            }
            if(previous > white_threshold)
            {
                continue;
            }
            if(left_white_seen
            && (current < black_threshold || is_gradient_edge(current, previous)))
            {
                left_edge = (uint16)col;
                left_detected = 1;
                image[row * IMG_COL + col] = 0;
                if(col + 1 < IMG_COL)
                {
                    image[row * IMG_COL + col + 1] = 0;
                }
                break;
            }
        }

        if(right_edge > left_edge
        && right_edge - left_edge >= MIN_VALID_TRACK_WIDTH
        && (left_detected || right_detected))
        {
            valid_rows++;
        }

        if(left_detected)
        {
            left_valid_count++;
            left_lost_run = 0;
        }
        else
        {
            left_lost_run++;
            if(left_lost_run > left_max_lost_run)
            {
                left_max_lost_run = left_lost_run;
            }
        }

        if(right_detected)
        {
            right_valid_count++;
            right_lost_run = 0;
        }
        else
        {
            right_lost_run++;
            if(right_lost_run > right_max_lost_run)
            {
                right_max_lost_run = right_lost_run;
            }
        }

        left_sum += left_edge;
        right_sum += right_edge;
        centre_sum += left_edge + right_edge;
    }

    /* Mark the reference white column for the LCD debug image. */
    for(row = IMG_ROW - 1; row >= boundary_row[reference_col]; row--)
    {
        image[row * IMG_COL + reference_col] = 0;
    }

    *line_valid = (uint8)(valid_rows >= MIN_VALID_SAMPLE_ROWS);
    *left_valid_count_output = left_valid_count;
    *right_valid_count_output = right_valid_count;
    *left_max_lost_run_output = left_max_lost_run;
    *right_max_lost_run_output = right_max_lost_run;
    *left_edge_average_output = (uint16)(left_sum / SAMPLE_ROW_COUNT);
    *right_edge_average_output = (uint16)(right_sum / SAMPLE_ROW_COUNT);
    return (uint16)(centre_sum / SAMPLE_ROW_COUNT);
}

void core1_main (void)
{
    disable_Watchdog();
#if LOW_COMPUTE_CPU1_INTERRUPT_WAKE
    /* Configure the request before the two cores pass their start barrier. */
    IfxSrc_clearRequest(&SRC_GPSR00);
    IfxSrc_init(&SRC_GPSR00,
                CPU1_FRAME_WAKE_INT_SERVICE,
                CPU1_FRAME_WAKE_INT_PRIO);
    IfxSrc_enable(&SRC_GPSR00);
#endif
    interrupt_global_enable(0);
    cpu_wait_event_ready();

    while(TRUE)
    {
        if(frame_ready)
        {
            uint8 valid;
            uint16 local_white_threshold;
            uint16 local_black_threshold;
            uint16 local_result;
            uint16 local_reference_col;
            uint16 local_reference_boundary_row;
            uint16 local_sample_start_row;
            uint8 local_reference_valid;
            uint8 local_left_valid_count;
            uint8 local_right_valid_count;
            uint8 local_left_max_lost_run;
            uint8 local_right_max_lost_run;
            uint16 local_left_edge_average;
            uint16 local_right_edge_average;

            __dsync();
            local_white_threshold = shared_tem_max;
            local_black_threshold = shared_tem_min;

            local_result = get_xian(img_buf,
                                    local_white_threshold,
                                    local_black_threshold,
                                    &valid,
                                    &local_reference_col,
                                    &local_reference_boundary_row,
                                    &local_sample_start_row,
                                    &local_reference_valid,
                                    &local_left_valid_count,
                                    &local_right_valid_count,
                                    &local_left_max_lost_run,
                                    &local_right_max_lost_run,
                                    &local_left_edge_average,
                                    &local_right_edge_average);

            z_result = local_result;
            line_valid_result = valid;
            reference_col_result = local_reference_col;
            reference_boundary_row_result = local_reference_boundary_row;
            sample_start_row_result = local_sample_start_row;
            reference_valid_result = local_reference_valid;
            left_valid_count_result = local_left_valid_count;
            right_valid_count_result = local_right_valid_count;
            left_max_lost_run_result = local_left_max_lost_run;
            right_max_lost_run_result = local_right_max_lost_run;
            left_edge_average_result = local_left_edge_average;
            right_edge_average_result = local_right_edge_average;
            frame_ready = 0;
            __dsync();
            result_ready = 1;
        }
#if LOW_COMPUTE_CPU1_INTERRUPT_WAKE
        else
        {
            /* Recheck immediately before IDLE to narrow the wake-up race. */
            __dsync();
            if(cpu1_idle_enabled && !frame_ready)
            {
                (void)IfxCpu_setCoreMode(&MODULE_CPU1,
                                         IfxCpu_CoreMode_idle);
            }
        }
#endif
    }
}

#if CPU1_PROCESSING_OPTIMIZE && defined(__TASKING__)
#pragma endoptimize
#endif

#pragma section all restore
