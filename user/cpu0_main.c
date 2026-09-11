/*********************************************************************************************************************
* TC264 Open Source Library
* Copyright (c) 2022 SEEKFREE Technology
*
* RT1064 smart-car application port for TC264D.
* The SEEKFREE library remains licensed under GPL-3.0-or-later.
********************************************************************************************************************/

#include "zf_common_headfile.h"
#include <stdlib.h>

#pragma section all "cpu0_dsram"

#define IPS200_TYPE                    (IPS200_TYPE_SPI)
#ifndef ENABLE_LCD_RUNTIME
#define ENABLE_LCD_RUNTIME             (0)
#endif

/* Start on the first real motor command and latch stopped after 20 seconds. */
#ifndef LAP_TIMED_STOP_ENABLE
#define LAP_TIMED_STOP_ENABLE          (0)
#endif
#ifndef LAP_TIMED_STOP_SECONDS
#define LAP_TIMED_STOP_SECONDS         (20U)
#endif

#if LAP_TIMED_STOP_ENABLE && (LAP_TIMED_STOP_SECONDS < 1U)
#error "LAP_TIMED_STOP_SECONDS must be at least 1"
#endif

#define IMG_ROW                        (120)
#define IMG_COL                        (188)
#define THRESHOLD_SAMPLE_ROWS          (2)
#define THRESHOLD_SAMPLE_HALF_WIDTH    (10)
#define THRESHOLD_BLACK_RATIO          (7)
#define THRESHOLD_WHITE_RATIO          (9)

/* Values copied from the RT1064 application. */
#define BASE_SPEED_PERCENT             (20)
#define OUTER_SPEED_GAIN_PERCENT       (25)
#define INNER_REVERSE_MAX_PERCENT      (35)
#define MAX_DUTY                       (50)
#define PID_OUTPUT_SCALE               (60.0f)
/* First-order derivative filter coefficient: 0 = no filtering, ->1 smoother. */
#define PID_D_FILTER_ALPHA             (0.5f)
#define STEERING_COMMAND_MAX           (BASE_SPEED_PERCENT + INNER_REVERSE_MAX_PERCENT)

#define CAMERA_SAFE_STOP_TIMEOUT_MS    (100)
#define CPU1_SAFE_STOP_TIMEOUT_MS      (50)

/* Longest-white-column reference. PID steering always remains at 100%. */
#define REFERENCE_CENTER_COL            (IMG_COL / 2)

/* Roundabout gap recognition. All values are deliberately easy to tune. */
#define EDGE_SAMPLE_COUNT               (10)
#define GAP_REFERENCE_CENTER_RANGE      (25)
#define GAP_SECOND_REFERENCE_CENTER_RANGE (25)
#define GAP_CENTER_SUM_TARGET           (IMG_COL)
#define GAP_CENTER_SUM_OFFSET_MIN       (25)
#define GAP_LOST_COUNT_MIN              (6)
#define GAP_LOST_RUN_MIN                (4)
#define GAP_OPPOSITE_VALID_MIN          (9)
#define GAP_CONFIRM_FRAMES              (3)
#define GAP_SECOND_CONFIRM_FRAMES       (2)
#define GAP_RELEASE_FRAMES              (3)
#define SECOND_GAP_TIMEOUT_REFERENCE_FRAMES (150)
#define SECOND_GAP_TIMEOUT_MIN_FRAMES   (70)
#define SECOND_GAP_TIMEOUT_MAX_FRAMES   (200)
#define FORCED_LEFT_EDGE_COL            (45)
#define FORCED_RIGHT_EDGE_COL           (143)
#define EDGE_BORDER_LOST_MARGIN         (6)
#define GAP_OPPOSITE_EDGE_TOLERANCE     (35)
#define PID_NEUTRAL_DEADBAND            (3)

/* Roundabout control and exit recognition. */
#define RING_ONLY_TEST                   (0)
#define RING_ENTRY_TURN_FRAMES          (20)
#define RING_SECOND_TURN_FRAMES         (10)
#define RING_FORCE_MIN_FRAMES           (5)
#define RING_BOUNDARY_ASSIST_FRAMES     (25)
#define RING_WAIT_REFERENCE_LOST_MAX    (12)
#define RING_EXIT_BLIND_FRAMES          (30)
#define RING_EXIT_TIMEOUT_FRAMES        (200)
#define RING_BOTH_LOST_VALID_MAX        (4)
#define RING_BOTH_LOST_CONFIRM_FRAMES   (1)
#define RING_TRACK_RECOVERY_VALID_MIN   (8)
#define RING_TRACK_RECOVERY_CONFIRM_FRAMES (2)
#define RING_FORCE_RATIO_PERCENT        (190)
#define RING_EXIT_FORCE_RATIO_PERCENT   (220)
#define RING_FORCE_SPEED_GAIN_PERCENT   (5)
#define RING_FORCE_RATIO_MIN            (100)
#define RING_FORCE_RATIO_MAX            (220)
#define RING_FORCE_MIN_COMMAND          (2)

/*
 * New-car motor wiring copied from the verified znc_new motor.h.
 * MOTOR1 uses the P02_4/P02_5 channel pair and MOTOR2 uses the
 * P02_6/P02_7 pair (drv8701e double-motor module, 17 kHz).
 *
 * Direction forward/reverse levels are taken from the E3_04
 * drv8701e double-motor demo and corrected on the bench:
 *   P02_4 side (MOTOR1/left): HIGH = forward, LOW = reverse.
 *   P02_6 side (MOTOR2/right): HIGH = forward, LOW = reverse.
 */
#define DIR_R1                         (P02_6)
#define PWM_R1                         (ATOM0_CH7_P02_7)
#define DIR_L1                         (P02_4)
#define PWM_L1                         (ATOM0_CH5_P02_5)

#define MOTOR1_PWM1                    (PWM_L1)
#define MOTOR1_PWM2                    (DIR_L1)
#define MOTOR2_PWM1                    (PWM_R1)
#define MOTOR2_PWM2                    (DIR_R1)

#define LEFT_FORWARD_LEVEL             (GPIO_HIGH)
#define LEFT_REVERSE_LEVEL             (GPIO_LOW)
#define RIGHT_FORWARD_LEVEL            (GPIO_HIGH)
#define RIGHT_REVERSE_LEVEL            (GPIO_LOW)

/* CPU0 writes the frame and thresholds. CPU1 writes the recognition result. */
uint8 img_buf[IMG_ROW * IMG_COL];
volatile uint8 frame_ready = 0;
volatile uint8 result_ready = 0;
#if LOW_COMPUTE_CPU1_INTERRUPT_WAKE
/* Cleared by CPU0 if the interrupt/IDLE path ever needs its RUN fallback. */
volatile uint8 cpu1_idle_enabled = 1;
#endif
volatile uint8 line_valid_result = 0;
volatile uint16 shared_tem_max = 0;
volatile uint16 shared_tem_min = 0;
volatile uint16 z_result = IMG_COL;
volatile uint16 reference_col_result = REFERENCE_CENTER_COL;
volatile uint16 reference_boundary_row_result = IMG_ROW - 1U;
volatile uint16 sample_start_row_result = IMG_ROW - 1U;
volatile uint8 reference_valid_result = 0;
volatile uint8 left_valid_count_result = 0;
volatile uint8 right_valid_count_result = 0;
volatile uint8 left_max_lost_run_result = 0;
volatile uint8 right_max_lost_run_result = 0;
volatile uint16 left_edge_average_result = 0;
volatile uint16 right_edge_average_result = IMG_COL - 1;

typedef struct
{
    float kp;
    float ki;
    float kd;
    float target;
    float current;
    float error;
    float error_last;
    float error_integral;
    float derivative_filtered;
    float output;
    float output_max;
    float output_min;
} pid_struct;

static pid_struct pid_pos =
{
    .kp = 45.0f,
    .ki = 0.0f,
    .kd =120.0f,
    .target = 188.0f,
    .output_max = PID_OUTPUT_SCALE * STEERING_COMMAND_MAX,
    .output_min = 0.0f
};

static uint8 turn_y = 0;
static uint8 turn_z = 0;

typedef enum
{
    RING_DIRECTION_NONE = 0,
    RING_DIRECTION_LEFT,
    RING_DIRECTION_RIGHT
} ring_direction_t;

typedef enum
{
    RING_STATE_NORMAL = 0,
    RING_STATE_WAIT_SECOND_GAP,
    RING_STATE_ENTRY_TURN,
    RING_STATE_INSIDE,
    RING_STATE_EXIT_TURN,
    RING_STATE_EXIT_TRACK
} ring_state_t;

static ring_state_t ring_state = RING_STATE_NORMAL;
static ring_direction_t ring_direction = RING_DIRECTION_NONE;
static ring_direction_t gap_candidate_direction = RING_DIRECTION_NONE;
static uint8 gap_candidate_frames = 0;
static uint8 gap_release_frames = GAP_RELEASE_FRAMES;
static uint8 gap_armed = 1;
static uint16 second_gap_wait_frames = 0;
static uint16 ring_age_frames = 0;
static uint8 ring_entry_turn_frames = 0;
static uint8 ring_second_turn_frames = 0;
static uint8 boundary_assist_frames = 0;
static uint8 both_lost_frames = 0;
static uint8 reference_lost_frames = 0;
static uint8 track_recovery_frames = 0;

#if LAP_TIMED_STOP_ENABLE
static uint32 lap_timer_ticks_per_second;
static uint64 lap_timer_limit_ticks;
static uint64 lap_timer_start_ticks;
static uint64 lap_timer_elapsed_ticks;
static uint8 lap_timer_started;
static uint8 lap_stop_latched;
#endif

static uint16 zhuanxiang = 0;

static int clamp_int (int value, int minimum, int maximum)
{
    if(value < minimum)
    {
        return minimum;
    }
    if(value > maximum)
    {
        return maximum;
    }
    return value;
}

/*
 * A transition found almost on the image border is not a reliable track
 * boundary. Lens shading and the camera-frame edge can otherwise make a
 * physically missing edge look valid on all ten sampled rows.
 */
static uint8 is_left_edge_lost (uint16 edge_average, uint8 valid_count)
{
    return (uint8)(valid_count <= RING_BOTH_LOST_VALID_MAX
                || edge_average <= EDGE_BORDER_LOST_MARGIN);
}

static uint8 is_right_edge_lost (uint16 edge_average, uint8 valid_count)
{
    return (uint8)(valid_count <= RING_BOTH_LOST_VALID_MAX
                || edge_average >= IMG_COL - 1U - EDGE_BORDER_LOST_MARGIN);
}

/* Require two trustworthy boundaries before a forced turn may end early. */
static uint8 are_both_edges_recovered (uint16 left_edge_average,
                                       uint16 right_edge_average,
                                       uint8 left_valid_count,
                                       uint8 right_valid_count)
{
    return (uint8)(
        left_valid_count >= RING_TRACK_RECOVERY_VALID_MIN
     && right_valid_count >= RING_TRACK_RECOVERY_VALID_MIN
     && !is_left_edge_lost(left_edge_average, left_valid_count)
     && !is_right_edge_lost(right_edge_average, right_valid_count)
     && right_edge_average > left_edge_average);
}

/*
 * Keep approximately the same travel distance between the two gap events.
 * Higher speed therefore gets fewer frames, and lower speed gets more frames.
 * BASE_SPEED_PERCENT uses the requested 80-frame reference point.
 */
static uint16 get_second_gap_timeout_frames (uint16 current_speed)
{
    int timeout_frames;

    if(current_speed == 0U)
    {
        return SECOND_GAP_TIMEOUT_MAX_FRAMES;
    }

    timeout_frames = SECOND_GAP_TIMEOUT_REFERENCE_FRAMES
                   * BASE_SPEED_PERCENT
                   / (int)current_speed;
    return (uint16)clamp_int(timeout_frames,
                             SECOND_GAP_TIMEOUT_MIN_FRAMES,
                             SECOND_GAP_TIMEOUT_MAX_FRAMES);
}

static ring_direction_t detect_gap_candidate (uint16 reference_col,
                                               uint8 reference_valid,
                                               uint16 centre_sum,
                                               uint16 left_edge_average,
                                               uint16 right_edge_average,
                                               uint8 left_valid_count,
                                               uint8 right_valid_count,
                                               uint8 left_max_lost_run,
                                               uint8 right_max_lost_run,
                                               uint16 reference_range)
{
    uint8 left_lost_count = EDGE_SAMPLE_COUNT - left_valid_count;
    uint8 right_lost_count = EDGE_SAMPLE_COUNT - right_valid_count;
    uint8 reference_near_center;
    uint8 left_missing_gap;
    uint8 right_missing_gap;
    uint8 left_geometry_gap;
    uint8 right_geometry_gap;
    uint8 left_gap;
    uint8 right_gap;
    uint8 left_edge_near_normal;
    uint8 right_edge_near_normal;

    reference_near_center = (uint8)(reference_valid
        && abs((int)reference_col - REFERENCE_CENTER_COL)
           <= reference_range);

    /*
     * A roundabout gap must still have a trustworthy opposite boundary near
     * its normal column. This rejects a wide white field where both physical
     * edges are outside the camera but the image border is detected as a line.
     */
    left_edge_near_normal = (uint8)(
        !is_left_edge_lost(left_edge_average, left_valid_count)
        && abs((int)left_edge_average - FORCED_LEFT_EDGE_COL)
           <= GAP_OPPOSITE_EDGE_TOLERANCE);

    right_edge_near_normal = (uint8)(
        !is_right_edge_lost(right_edge_average, right_valid_count)
        && abs((int)right_edge_average - FORCED_RIGHT_EDGE_COL)
           <= GAP_OPPOSITE_EDGE_TOLERANCE);

    left_missing_gap = (uint8)(reference_near_center
        && left_lost_count >= GAP_LOST_COUNT_MIN
        && left_max_lost_run >= GAP_LOST_RUN_MIN
        && right_valid_count >= GAP_OPPOSITE_VALID_MIN
        && right_edge_near_normal);

    right_missing_gap = (uint8)(reference_near_center
        && right_lost_count >= GAP_LOST_COUNT_MIN
        && right_max_lost_run >= GAP_LOST_RUN_MIN
        && left_valid_count >= GAP_OPPOSITE_VALID_MIN
        && left_edge_near_normal);

    /*
     * At the real roundabout the inner circular arc can be mistaken for a
     * normal boundary, so L/R may both remain 10 and the lost-edge rule alone
     * never fires.  The videos show that the longest-white band remains near
     * the image centre while the average edge sum moves strongly toward the
     * opening.  Use that geometry as symmetric supplementary evidence.
     */
    left_geometry_gap = (uint8)(reference_near_center
        && centre_sum + GAP_CENTER_SUM_OFFSET_MIN <= GAP_CENTER_SUM_TARGET
        && right_valid_count >= GAP_OPPOSITE_VALID_MIN
        && right_edge_near_normal);

    right_geometry_gap = (uint8)(reference_near_center
        && centre_sum >= GAP_CENTER_SUM_TARGET + GAP_CENTER_SUM_OFFSET_MIN
        && left_valid_count >= GAP_OPPOSITE_VALID_MIN
        && left_edge_near_normal);

    left_gap = (uint8)(left_missing_gap || left_geometry_gap);
    right_gap = (uint8)(right_missing_gap || right_geometry_gap);

    if(left_gap && !right_gap)
    {
        return RING_DIRECTION_LEFT;
    }
    if(right_gap && !left_gap)
    {
        return RING_DIRECTION_RIGHT;
    }
    return RING_DIRECTION_NONE;
}

/*
 * Replace the missing roundabout-side boundary with the requested fixed
 * column. If the opposite boundary is temporarily also absent, use its normal
 * symmetric column so PID remains neutral until a real opposite edge returns.
 */
static uint16 get_boundary_assisted_z (ring_direction_t direction,
                                       uint16 left_edge_average,
                                       uint16 right_edge_average,
                                       uint8 left_valid_count,
                                       uint8 right_valid_count)
{
    uint8 left_edge_lost = is_left_edge_lost(left_edge_average,
                                              left_valid_count);
    uint8 right_edge_lost = is_right_edge_lost(right_edge_average,
                                                right_valid_count);

    if(direction == RING_DIRECTION_LEFT)
    {
        left_edge_average = FORCED_LEFT_EDGE_COL;
        if(right_edge_lost)
        {
            right_edge_average = FORCED_RIGHT_EDGE_COL;
        }
        return (uint16)(left_edge_average + right_edge_average);
    }

    if(direction == RING_DIRECTION_RIGHT)
    {
        right_edge_average = FORCED_RIGHT_EDGE_COL;
        if(left_edge_lost)
        {
            left_edge_average = FORCED_LEFT_EDGE_COL;
        }
        return (uint16)(left_edge_average + right_edge_average);
    }

    return (uint16)(left_edge_average + right_edge_average);
}

/* Convert a multi-frame gap into one edge-triggered event. */
static ring_direction_t update_gap_event (ring_direction_t candidate,
                                           uint8 confirm_frames)
{
    ring_direction_t event = RING_DIRECTION_NONE;

    if(candidate == RING_DIRECTION_NONE)
    {
        gap_candidate_direction = RING_DIRECTION_NONE;
        gap_candidate_frames = 0;
        if(gap_release_frames < GAP_RELEASE_FRAMES)
        {
            gap_release_frames++;
        }
        if(gap_release_frames >= GAP_RELEASE_FRAMES)
        {
            gap_armed = 1;
        }
        return event;
    }

    gap_release_frames = 0;
    if(candidate != gap_candidate_direction)
    {
        gap_candidate_direction = candidate;
        gap_candidate_frames = 1;
    }
    else if(gap_candidate_frames < confirm_frames)
    {
        gap_candidate_frames++;
    }

    if(gap_armed && gap_candidate_frames >= confirm_frames)
    {
        event = gap_candidate_direction;
        gap_armed = 0;
    }
    return event;
}

static void reset_gap_detector (void)
{
    gap_candidate_direction = RING_DIRECTION_NONE;
    gap_candidate_frames = 0;
    gap_release_frames = GAP_RELEASE_FRAMES;
    gap_armed = 1;
}

static void reset_ring_control (void)
{
    ring_state = RING_STATE_NORMAL;
    ring_direction = RING_DIRECTION_NONE;
    second_gap_wait_frames = 0;
    ring_age_frames = 0;
    ring_entry_turn_frames = 0;
    ring_second_turn_frames = 0;
    boundary_assist_frames = 0;
    both_lost_frames = 0;
    reference_lost_frames = 0;
    track_recovery_frames = 0;
    reset_gap_detector();
}

static uint32 percent_to_duty (int percent)
{
    percent = clamp_int(percent, 0, MAX_DUTY);
    return (uint32)percent * (PWM_DUTY_MAX / 100U);
}

static void motor_stop (void)
{
    pwm_set_duty(MOTOR1_PWM1, 0);
    pwm_set_duty(MOTOR2_PWM1, 0);
}

static void motor_set_forward (void)
{
    gpio_set_level(MOTOR1_PWM2, LEFT_FORWARD_LEVEL);
    gpio_set_level(MOTOR2_PWM2, RIGHT_FORWARD_LEVEL);
}

/*
 * Apply the RT1064 differential-drive mixer.
 * A hard turn reverses only the inside wheel. The new-car pin mapping and the
 * verified forward/reverse levels above remain the only hardware adaptation.
 */
static void motor_apply_steering_at_speed (int command,
                                           int base_speed_percent)
{
    int left_percent;
    int right_percent;
    uint32 left_duty;
    uint32 right_duty;

    base_speed_percent = clamp_int(base_speed_percent, 0, MAX_DUTY);
    left_percent = base_speed_percent;
    right_percent = base_speed_percent;
    command = clamp_int(command, 0, STEERING_COMMAND_MAX);
    motor_set_forward();

    if(turn_y)
    {
        /* Physical left turn: right wheel is outer, left wheel is inner. */
        right_percent = base_speed_percent
                      + clamp_int(command, 0, OUTER_SPEED_GAIN_PERCENT);

        if(command >= base_speed_percent)
        {
            gpio_set_level(MOTOR1_PWM2, LEFT_REVERSE_LEVEL);
            left_percent = clamp_int(command - base_speed_percent,
                                     0,
                                     INNER_REVERSE_MAX_PERCENT);
        }
        else
        {
            left_percent = base_speed_percent - command;
        }
    }
    else if(turn_z)
    {
        /* Physical right turn: left wheel is outer, right wheel is inner. */
        left_percent = base_speed_percent
                     + clamp_int(command, 0, OUTER_SPEED_GAIN_PERCENT);

        if(command >= base_speed_percent)
        {
            gpio_set_level(MOTOR2_PWM2, RIGHT_REVERSE_LEVEL);
            right_percent = clamp_int(command - base_speed_percent,
                                      0,
                                      INNER_REVERSE_MAX_PERCENT);
        }
        else
        {
            right_percent = base_speed_percent - command;
        }
    }

    left_percent = clamp_int(left_percent, 0, MAX_DUTY);
    right_percent = clamp_int(right_percent, 0, MAX_DUTY);
    left_duty = percent_to_duty(left_percent);
    right_duty = percent_to_duty(right_percent);

    pwm_set_duty(MOTOR1_PWM1, left_duty);
    pwm_set_duty(MOTOR2_PWM1, right_duty);
}

static int get_ring_force_command (int current_speed, int force_ratio)
{
    int command;
    int force_percent;

    current_speed = clamp_int(current_speed, 0, MAX_DUTY);
    if(current_speed <= 1)
    {
        return 0;
    }

    /* The faster the car, the sharper the forced turn must be. */
    force_percent = force_ratio
                  + (current_speed - BASE_SPEED_PERCENT)
                    * RING_FORCE_SPEED_GAIN_PERCENT;
    force_percent = clamp_int(force_percent,
                              RING_FORCE_RATIO_MIN,
                              RING_FORCE_RATIO_MAX);

    command = current_speed * force_percent / 100;
    if(command < RING_FORCE_MIN_COMMAND)
    {
        command = RING_FORCE_MIN_COMMAND;
    }
    return command;
}

static int motor_apply_ring_turn (ring_direction_t direction,
                                  int current_speed,
                                  int force_ratio)
{
    int command = get_ring_force_command(current_speed, force_ratio);

    turn_y = 0;
    turn_z = 0;

    /*
     * On the new car, turn_y speeds up the physical right wheel and
     * slows/reverses the physical left wheel, so it remains a left turn.
     * turn_z applies the mirrored physical right turn.
     */
    if(direction == RING_DIRECTION_LEFT)
    {
        turn_y = 1;
    }
    else if(direction == RING_DIRECTION_RIGHT)
    {
        turn_z = 1;
    }

    if(direction == RING_DIRECTION_NONE)
    {
        motor_stop();
        command = 0;
    }
    else
    {
        motor_apply_steering_at_speed(command, current_speed);
    }
    return command;
}

/* Average the same bottom-centre 2 x 20 ROI used by the RT1064 code. */
static uint16 get_dian (const uint8 *image)
{
    const uint16 start_y = IMG_ROW - THRESHOLD_SAMPLE_ROWS;
    const uint16 start_x = IMG_COL / 2U - THRESHOLD_SAMPLE_HALF_WIDTH;
    const uint16 total_pixels = THRESHOLD_SAMPLE_ROWS
                              * 2U
                              * THRESHOLD_SAMPLE_HALF_WIDTH;
    uint32 sum = 0;
    uint16 y;
    uint16 x;
    uint16 average;
    static uint16 first_average = 0;
    static uint8 first_frame = 1;

    for(y = 0; y < THRESHOLD_SAMPLE_ROWS; y++)
    {
        for(x = 0; x < 2U * THRESHOLD_SAMPLE_HALF_WIDTH; x++)
        {
            sum += image[(start_y + y) * IMG_COL + start_x + x];
        }
    }

    average = (uint16)(sum / total_pixels);
    if(first_frame)
    {
        first_average = average;
        first_frame = 0;
    }

    if(average < first_average * 8U / 10U)
    {
        average = first_average * 8U / 10U;
    }
    return average;
}

static float pid_calc (pid_struct *pid,
                       float current_value,
                       int8 *turn_direction)
{
    float derivative;

    pid->current = current_value;
    if(pid->target >= pid->current)
    {
        pid->error = pid->target - pid->current;
        if(turn_direction != 0)
        {
            *turn_direction = 1;
        }
    }
    else
    {
        pid->error = pid->current - pid->target;
        if(turn_direction != 0)
        {
            *turn_direction = -1;
        }
    }

    /*
     * Accumulate a true integral with anti-windup. The integral contribution is
     * clamped to the output range so it cannot wind up while saturated.
     */
    if(pid->ki != 0.0f)
    {
        pid->error_integral += pid->error;
        if(pid->error_integral > pid->output_max / pid->ki)
        {
            pid->error_integral = pid->output_max / pid->ki;
        }
        else if(pid->error_integral < pid->output_min / pid->ki)
        {
            pid->error_integral = pid->output_min / pid->ki;
        }
    }
    else
    {
        pid->error_integral = 0.0f;
    }

    /* First-order low-pass on the derivative term to reject image noise. */
    derivative = pid->error - pid->error_last;
    pid->error_last = pid->error;
    pid->derivative_filtered = PID_D_FILTER_ALPHA * pid->derivative_filtered
                             + (1.0f - PID_D_FILTER_ALPHA) * derivative;

    pid->output = pid->kp * pid->error
                + pid->ki * pid->error_integral
                + pid->kd * pid->derivative_filtered;

    /*
     * The RT1064 source declared this function as uint8 even though it returns
     * a float. Clamp the intended non-negative steering magnitude explicitly
     * instead of relying on target-specific float-to-uint8 conversion.
     */
    if(pid->output < pid->output_min)
    {
        pid->output = pid->output_min;
    }
    if(pid->output > pid->output_max)
    {
        pid->output = pid->output_max;
    }
    return pid->output;
}

static void pid_reset (pid_struct *pid)
{
    pid->error = 0.0f;
    pid->error_last = 0.0f;
    pid->error_integral = 0.0f;
    pid->derivative_filtered = 0.0f;
    pid->output = 0.0f;
}

/*
 * Keep the PID history aligned while a forced roundabout turn owns the motors.
 * This prevents a derivative kick when normal PID tracking takes control again.
 */
static void pid_track_without_output (pid_struct *pid, float current_value)
{
    pid->current = current_value;
    if(pid->target >= pid->current)
    {
        pid->error = pid->target - pid->current;
    }
    else
    {
        pid->error = pid->current - pid->target;
    }

    pid->error_last = pid->error;
    pid->error_integral = 0.0f;
    pid->derivative_filtered = 0.0f;
    pid->output = 0.0f;
}

static void update_ring_state (ring_direction_t gap_event,
                               uint8 both_edges_lost,
                               uint8 both_edges_recovered,
                               uint8 reference_near_center,
                               uint16 second_gap_timeout_frames)
{
    switch(ring_state)
    {
        case RING_STATE_NORMAL:
        {
            if(gap_event != RING_DIRECTION_NONE)
            {
                ring_direction = gap_event;
                second_gap_wait_frames = 0;
                reference_lost_frames = 0;
                ring_state = RING_STATE_WAIT_SECOND_GAP;
            }
            break;
        }

        case RING_STATE_WAIT_SECOND_GAP:
        {
            if(reference_near_center)
            {
                reference_lost_frames = 0;
            }
            else if(reference_lost_frames < RING_WAIT_REFERENCE_LOST_MAX)
            {
                reference_lost_frames++;
            }

            /* Allow brief body oscillation, but reject a sustained departure. */
            if(reference_lost_frames >= RING_WAIT_REFERENCE_LOST_MAX)
            {
                reset_ring_control();
                break;
            }

            if(second_gap_wait_frames < second_gap_timeout_frames)
            {
                second_gap_wait_frames++;
            }

            if(gap_event != RING_DIRECTION_NONE)
            {
                if(gap_event == ring_direction)
                {
                    ring_entry_turn_frames = 0;
                    ring_age_frames = 0;
                    both_lost_frames = 0;
                    ring_second_turn_frames = 0;
                    boundary_assist_frames = 0;
                    reference_lost_frames = 0;
                    track_recovery_frames = 0;
                    zhuanxiang++;
                    ring_state = RING_STATE_ENTRY_TURN;
                }
                else
                {
                    /* The opposite gap becomes a new first-gap observation. */
                    ring_direction = gap_event;
                    second_gap_wait_frames = 0;
                    reference_lost_frames = 0;
                }
            }
            else if(second_gap_wait_frames >= second_gap_timeout_frames)
            {
                ring_state = RING_STATE_NORMAL;
                ring_direction = RING_DIRECTION_NONE;
                second_gap_wait_frames = 0;
            }
            break;
        }

        case RING_STATE_ENTRY_TURN:
        {
            if(ring_entry_turn_frames < RING_ENTRY_TURN_FRAMES)
            {
                ring_entry_turn_frames++;
            }
            if(ring_age_frames < RING_EXIT_TIMEOUT_FRAMES + 1U)
            {
                ring_age_frames++;
            }

            if(both_edges_recovered)
            {
                if(track_recovery_frames
                   < RING_TRACK_RECOVERY_CONFIRM_FRAMES)
                {
                    track_recovery_frames++;
                }
            }
            else
            {
                track_recovery_frames = 0;
            }

            if((ring_entry_turn_frames >= RING_FORCE_MIN_FRAMES
             && track_recovery_frames
                >= RING_TRACK_RECOVERY_CONFIRM_FRAMES)
            || ring_entry_turn_frames >= RING_ENTRY_TURN_FRAMES)
            {
                track_recovery_frames = 0;
                ring_state = RING_STATE_INSIDE;
            }
            break;
        }

        case RING_STATE_INSIDE:
        {
            if(ring_age_frames < RING_EXIT_TIMEOUT_FRAMES + 1U)
            {
                ring_age_frames++;
            }

            if(ring_age_frames >= RING_EXIT_BLIND_FRAMES
            && ring_age_frames <= RING_EXIT_TIMEOUT_FRAMES
            && both_edges_lost)
            {
                if(both_lost_frames < RING_BOTH_LOST_CONFIRM_FRAMES)
                {
                    both_lost_frames++;
                }
            }
            else
            {
                both_lost_frames = 0;
            }

            if(both_lost_frames >= RING_BOTH_LOST_CONFIRM_FRAMES)
            {
                ring_second_turn_frames = 0;
                boundary_assist_frames = 0;
                track_recovery_frames = 0;
                zhuanxiang++;
                ring_state = RING_STATE_EXIT_TURN;
            }
            else if(ring_age_frames > RING_EXIT_TIMEOUT_FRAMES)
            {
                /* A timeout without both-edge loss is not a valid exit. */
                reset_ring_control();
            }
            break;
        }

        case RING_STATE_EXIT_TURN:
        {
            if(ring_second_turn_frames < RING_SECOND_TURN_FRAMES)
            {
                ring_second_turn_frames++;
            }

            if(both_edges_recovered)
            {
                if(track_recovery_frames
                   < RING_TRACK_RECOVERY_CONFIRM_FRAMES)
                {
                    track_recovery_frames++;
                }
            }
            else
            {
                track_recovery_frames = 0;
            }

            if((ring_second_turn_frames >= RING_FORCE_MIN_FRAMES
             && track_recovery_frames
                >= RING_TRACK_RECOVERY_CONFIRM_FRAMES)
            || ring_second_turn_frames >= RING_SECOND_TURN_FRAMES)
            {
                boundary_assist_frames = 0;
                track_recovery_frames = 0;
                ring_state = RING_STATE_EXIT_TRACK;
            }
            break;
        }

        case RING_STATE_EXIT_TRACK:
        {
            if(boundary_assist_frames < RING_BOUNDARY_ASSIST_FRAMES)
            {
                boundary_assist_frames++;
            }

            if(boundary_assist_frames >= RING_BOUNDARY_ASSIST_FRAMES)
            {
                reset_ring_control();
            }
            break;
        }

        default:
        {
            reset_ring_control();
            break;
        }
    }
}

/* Keep waiting for recovery, but make a stalled pipeline physically safe. */
static void wait_flag_with_safe_stop (volatile uint8 *flag, uint16 timeout_ms)
{
    uint16 elapsed_ms = 0;
    uint8 stopped = 0;

    while(!(*flag))
    {
        system_delay_ms(1);
        if(elapsed_ms < timeout_ms)
        {
            elapsed_ms++;
        }
        else if(!stopped)
        {
            motor_stop();
            pid_reset(&pid_pos);
            stopped = 1;
        }
    }
}

#if LOW_COMPUTE_CPU1_INTERRUPT_WAKE
/*
 * Normal path: GPSR00 wakes CPU1 immediately.
 * Fallback path: while retaining the original 1 ms wait and 50 ms safe-stop,
 * CPU0 periodically verifies CPU1's mode. If a request was missed during the
 * RUN-to-IDLE transition, CPU0 requests the interrupt again and explicitly
 * returns CPU1 to RUN. A broken wake request therefore cannot leave CPU1
 * permanently idle.
 */
static void wait_cpu1_result_with_interrupt_fallback (void)
{
    uint16 elapsed_ms = 0;
    uint16 fallback_elapsed_ms = 0;
    uint8 stopped = 0;

    while(!result_ready)
    {
        system_delay_ms(1);

        if(elapsed_ms < CPU1_SAFE_STOP_TIMEOUT_MS)
        {
            elapsed_ms++;
        }
        else if(!stopped)
        {
            motor_stop();
            pid_reset(&pid_pos);
            /* Unknown/failed interrupt state: restore proven polling mode. */
            cpu1_idle_enabled = 0;
            __dsync();
            IfxSrc_setRequest(&SRC_GPSR00);
            (void)IfxCpu_setCoreMode(&MODULE_CPU1, IfxCpu_CoreMode_run);
            stopped = 1;
        }

        if(!result_ready)
        {
            fallback_elapsed_ms++;
            if(fallback_elapsed_ms >= CPU1_WAKE_FALLBACK_CHECK_MS)
            {
                fallback_elapsed_ms = 0;
                __dsync();
                if(frame_ready
                && IfxCpu_getCoreMode(&MODULE_CPU1)
                    == IfxCpu_CoreMode_idle)
                {
                    /* One missed wake permanently falls back to known polling. */
                    cpu1_idle_enabled = 0;
                    __dsync();
                    IfxSrc_setRequest(&SRC_GPSR00);
                    __dsync();
                    if(IfxCpu_getCoreMode(&MODULE_CPU1)
                        == IfxCpu_CoreMode_idle)
                    {
                        (void)IfxCpu_setCoreMode(&MODULE_CPU1,
                                                 IfxCpu_CoreMode_run);
                    }
                }
            }
        }
    }
}
#endif

void core0_main (void)
{
    uint16 gray_average;
    uint16 current_z;
    uint16 control_z;
    uint16 current_reference_col;
    uint16 current_left_edge_average;
    uint16 current_right_edge_average;
    uint16 image_base_speed_percent;
    uint16 current_second_gap_timeout_frames;
    uint16 current_gap_reference_range;
    uint8 current_gap_confirm_frames;
    uint8 current_line_valid;
    uint8 current_reference_valid;
    uint8 current_left_valid_count;
    uint8 current_right_valid_count;
    uint8 current_left_max_lost_run;
    uint8 current_right_max_lost_run;
    uint8 both_edges_lost;
    uint8 both_edges_recovered;
    uint8 reference_near_center;
    uint8 boundary_assist_active;
#if !RING_ONLY_TEST
    float steering_output;
    int8 steering_turn;
#endif
    int steering_command;
    int ring_turn_command;
    uint8 ring_force_active;
#if LAP_TIMED_STOP_ENABLE
    uint8 motor_command_active;
#endif
    ring_direction_t gap_candidate;
    ring_direction_t gap_event;
    ring_state_t previous_ring_state;

    clock_init();
#if LAP_TIMED_STOP_ENABLE
    lap_timer_ticks_per_second =
        (uint32)IfxStm_getFrequency(&MODULE_STM0);
    lap_timer_limit_ticks = (uint64)lap_timer_ticks_per_second
                          * (uint64)LAP_TIMED_STOP_SECONDS;
#endif
    debug_init();
    system_delay_ms(300);

#if ENABLE_LCD_RUNTIME
    ips200_init(IPS200_TYPE);
#endif
    /* Preserve the original startup timing even when the LCD is disabled. */
    system_delay_ms(300);
#if ENABLE_LCD_RUNTIME
    ips200_show_string(0, 0, "mt9v03x init.");
#endif

    gpio_init(DIR_R1, GPO, RIGHT_FORWARD_LEVEL, GPO_PUSH_PULL);
    pwm_init(PWM_R1, 17000, 0);
    gpio_init(DIR_L1, GPO, LEFT_FORWARD_LEVEL, GPO_PUSH_PULL);
    pwm_init(PWM_L1, 17000, 0);
    motor_set_forward();
    motor_stop();

    /* TC264 uses zero as the argument that enables global interrupts. */
    interrupt_global_enable(0);
    while(mt9v03x_init())
    {
#if ENABLE_LCD_RUNTIME
        ips200_show_string(0, 16, "mt9v03x reinit.");
#endif
        system_delay_ms(500);
    }

#if ENABLE_LCD_RUNTIME
    ips200_show_string(0, 16, "init success.");
#endif
    (void)mt9v03x_set_exposure_time(300);

    cpu_wait_event_ready();
    pid_reset(&pid_pos);
    reset_ring_control();

    while(1)
    {
        wait_flag_with_safe_stop(&mt9v03x_finish_flag,
                                 CAMERA_SAFE_STOP_TIMEOUT_MS);
        mt9v03x_finish_flag = 0;
        memcpy(img_buf, mt9v03x_image, sizeof(img_buf));

        gray_average = get_dian(img_buf);
        shared_tem_min = gray_average * THRESHOLD_BLACK_RATIO / 10U;
        shared_tem_max = gray_average * THRESHOLD_WHITE_RATIO / 10U;

        result_ready = 0;
        __dsync();
        frame_ready = 1;
#if LOW_COMPUTE_CPU1_INTERRUPT_WAKE
        __dsync();
        IfxSrc_setRequest(&SRC_GPSR00);
#endif

        /* Use the current frame, matching the single-core RT1064 program. */
#if LOW_COMPUTE_CPU1_INTERRUPT_WAKE
        wait_cpu1_result_with_interrupt_fallback();
#else
        wait_flag_with_safe_stop(&result_ready, CPU1_SAFE_STOP_TIMEOUT_MS);
#endif
        __dsync();
        current_z = z_result;
        current_line_valid = line_valid_result;
        current_reference_col = reference_col_result;
        current_reference_valid = reference_valid_result;
        current_left_valid_count = left_valid_count_result;
        current_right_valid_count = right_valid_count_result;
        current_left_max_lost_run = left_max_lost_run_result;
        current_right_max_lost_run = right_max_lost_run_result;
        current_left_edge_average = left_edge_average_result;
        current_right_edge_average = right_edge_average_result;
        result_ready = 0;

        current_gap_reference_range = GAP_REFERENCE_CENTER_RANGE;
        current_gap_confirm_frames = GAP_CONFIRM_FRAMES;
        if(ring_state == RING_STATE_WAIT_SECOND_GAP)
        {
            current_gap_reference_range = GAP_SECOND_REFERENCE_CENTER_RANGE;
            current_gap_confirm_frames = GAP_SECOND_CONFIRM_FRAMES;
        }

        gap_candidate = detect_gap_candidate(current_reference_col,
                                              current_reference_valid,
                                              current_z,
                                              current_left_edge_average,
                                              current_right_edge_average,
                                              current_left_valid_count,
                                              current_right_valid_count,
                                              current_left_max_lost_run,
                                              current_right_max_lost_run,
                                              current_gap_reference_range);
        gap_event = update_gap_event(gap_candidate, current_gap_confirm_frames);

        reference_near_center = (uint8)(current_reference_valid
            && abs((int)current_reference_col - REFERENCE_CENTER_COL)
               <= GAP_REFERENCE_CENTER_RANGE);
        image_base_speed_percent = BASE_SPEED_PERCENT;
        current_second_gap_timeout_frames = get_second_gap_timeout_frames(
            image_base_speed_percent);

        both_edges_lost = (uint8)(
            is_left_edge_lost(current_left_edge_average,
                              current_left_valid_count)
         && is_right_edge_lost(current_right_edge_average,
                               current_right_valid_count));

        both_edges_recovered = are_both_edges_recovered(
            current_left_edge_average,
            current_right_edge_average,
            current_left_valid_count,
            current_right_valid_count);

        previous_ring_state = ring_state;
        update_ring_state(gap_event,
                          both_edges_lost,
                          both_edges_recovered,
                          reference_near_center,
                          current_second_gap_timeout_frames);

        if(previous_ring_state != ring_state)
        {
            if(ring_state == RING_STATE_ENTRY_TURN
            || ring_state == RING_STATE_EXIT_TURN
            || ring_state == RING_STATE_NORMAL)
            {
                pid_reset(&pid_pos);
            }

        }

        turn_y = 0;
        turn_z = 0;
        ring_force_active = 0;
        boundary_assist_active = 0;
        steering_command = 0;
        ring_turn_command = 0;
#if LAP_TIMED_STOP_ENABLE
        motor_command_active = 0;
#endif
        control_z = current_z;

        if(ring_state == RING_STATE_WAIT_SECOND_GAP)
        {
            boundary_assist_active = 1;
            control_z = get_boundary_assisted_z(
                ring_direction,
                current_left_edge_average,
                current_right_edge_average,
                current_left_valid_count,
                current_right_valid_count);
        }
        else if(ring_state == RING_STATE_EXIT_TURN
        || ring_state == RING_STATE_EXIT_TRACK
        || (ring_state == RING_STATE_INSIDE
         && both_edges_lost
         && ring_age_frames >= RING_EXIT_BLIND_FRAMES))
        {
            boundary_assist_active = 1;
            control_z = get_boundary_assisted_z(
                ring_direction,
                current_left_edge_average,
                current_right_edge_average,
                current_left_valid_count,
                current_right_valid_count);
        }
        else if(ring_state == RING_STATE_NORMAL
             && gap_candidate != RING_DIRECTION_NONE)
        {
            boundary_assist_active = 1;
            control_z = get_boundary_assisted_z(
                gap_candidate,
                current_left_edge_average,
                current_right_edge_average,
                current_left_valid_count,
                current_right_valid_count);
        }

        /* Remove the harmless 1-3 count offset around the calibrated target. */
        if(abs((int)control_z - (int)pid_pos.target)
           <= PID_NEUTRAL_DEADBAND)
        {
            control_z = (uint16)pid_pos.target;
        }

#if LAP_TIMED_STOP_ENABLE
        /*
         * Check the 64-bit STM before writing any new PWM command.  This also
         * prevents a delayed camera/CPU1 result from briefly restarting the
         * motors after the 20-second deadline.
         */
        if(lap_timer_started && !lap_stop_latched)
        {
            lap_timer_elapsed_ticks = IfxStm_get(&MODULE_STM0)
                                    - lap_timer_start_ticks;
            if(lap_timer_elapsed_ticks >= lap_timer_limit_ticks)
            {
                lap_timer_elapsed_ticks = lap_timer_limit_ticks;
                lap_stop_latched = 1;
            }
        }

        if(lap_stop_latched)
        {
            motor_stop();
            pid_reset(&pid_pos);
        }
        else
#endif
        if(ring_state == RING_STATE_ENTRY_TURN
        || ring_state == RING_STATE_EXIT_TURN)
        {
            ring_force_active = 1;
            pid_track_without_output(&pid_pos, (float)control_z);
            ring_turn_command = motor_apply_ring_turn(
                ring_direction,
                image_base_speed_percent,
                (ring_state == RING_STATE_EXIT_TURN)
                    ? RING_EXIT_FORCE_RATIO_PERCENT
                    : RING_FORCE_RATIO_PERCENT);
#if LAP_TIMED_STOP_ENABLE
            motor_command_active = (uint8)(ring_turn_command > 0);
#endif
        }
        else if(ring_state == RING_STATE_INSIDE && both_edges_lost)
        {
            /* Keep moving while two-edge loss is confirmed; do not stop. */
            pid_track_without_output(&pid_pos, (float)control_z);
            motor_apply_steering_at_speed(0, image_base_speed_percent);
#if LAP_TIMED_STOP_ENABLE
            motor_command_active = 1;
#endif
        }
        else if(current_line_valid || boundary_assist_active)
        {
#if RING_ONLY_TEST
            /*
             * Roundabout-only bench mode: keep both wheels at base speed and
             * completely remove the image PID from the motor command. Entry
             * and exit turns above still use the speed-related ring command.
             */
            pid_reset(&pid_pos);
            motor_apply_steering_at_speed(0, image_base_speed_percent);
#else
            steering_output = pid_calc(&pid_pos, (float)control_z,
                                       &steering_turn)
                            / PID_OUTPUT_SCALE;
            turn_y = (steering_turn > 0) ? 1 : 0;
            turn_z = (steering_turn < 0) ? 1 : 0;
            /* Longest-white position no longer scales PID: apply full output. */
            steering_command = (int)(steering_output + 0.5f);
            motor_apply_steering_at_speed(steering_command,
                                          image_base_speed_percent);
#endif
#if LAP_TIMED_STOP_ENABLE
            motor_command_active = 1;
#endif
        }
        else
        {
            motor_stop();
            pid_reset(&pid_pos);
        }

#if LAP_TIMED_STOP_ENABLE
        if(!lap_timer_started && motor_command_active)
        {
            lap_timer_start_ticks = IfxStm_get(&MODULE_STM0);
            lap_timer_elapsed_ticks = 0;
            lap_timer_started = 1;
        }
#endif

#if !ENABLE_LCD_RUNTIME
        /* These original values only feed the disabled LCD diagnostics. */
        (void)ring_force_active;
        (void)ring_turn_command;
#endif

#if ENABLE_LCD_RUNTIME
        ips200_show_gray_image(0,
                               0,
                               (const uint8 *)img_buf,
                               MT9V03X_W,
                               MT9V03X_H,
                               240,
                               180,
                               0);

        /*
         * Roundabout diagnostics below the 240 x 180 camera image.
         * Direction values: 0=none, 1=left, 2=right.
         */
        ips200_set_color(RGB565_RED, RGB565_BLACK);
        ips200_show_string(0, 184, "T:");
        ips200_show_uint(16, 184, RING_ONLY_TEST, 1);
        ips200_show_string(32, 184, "S:");
        ips200_show_uint(48, 184, (uint32)ring_state, 1);
        ips200_show_string(64, 184, "D:");
        ips200_show_uint(80, 184, (uint32)ring_direction, 1);
        ips200_show_string(96, 184, "F:");
        ips200_show_uint(112, 184, ring_force_active, 1);
        ips200_show_string(136, 184, "LV:");
        ips200_show_uint(160, 184, current_line_valid, 1);
        ips200_show_string(184, 184, "RV:");
        ips200_show_uint(208, 184, current_reference_valid, 1);

        ips200_show_string(0, 200, "GC:");
        ips200_show_uint(24, 200, (uint32)gap_candidate, 1);
        ips200_show_string(48, 200, "GE:");
        ips200_show_uint(72, 200, (uint32)gap_event, 1);
        ips200_show_string(96, 200, "CF:");
        ips200_show_uint(120, 200, gap_candidate_frames, 1);
        ips200_show_string(144, 200, "AR:");
        ips200_show_uint(168, 200, gap_armed, 1);
        ips200_show_string(192, 200, "RF:");
        ips200_show_uint(216, 200, gap_release_frames, 1);

        ips200_show_string(0, 216, "C:");
        ips200_show_uint(16, 216, current_reference_col, 3);
        ips200_show_string(56, 216, "Z:");
        ips200_show_uint(72, 216, current_z, 3);
        ips200_show_string(112, 216, "W:");
        ips200_show_uint(128, 216, second_gap_wait_frames, 3);
        ips200_show_string(176, 216, "A:");
        ips200_show_uint(192, 216, ring_age_frames, 3);

        ips200_show_string(0, 232, "L:");
        ips200_show_uint(16, 232, current_left_valid_count, 2);
        ips200_show_string(48, 232, "R:");
        ips200_show_uint(64, 232, current_right_valid_count, 2);
        ips200_show_string(96, 232, "LL:");
        ips200_show_uint(120, 232, current_left_max_lost_run, 2);
        ips200_show_string(152, 232, "RL:");
        ips200_show_uint(176, 232, current_right_max_lost_run, 2);

        ips200_show_string(0, 248, "E:");
        ips200_show_uint(16, 248, ring_entry_turn_frames, 2);
        ips200_show_string(48, 248, "ST:");
        ips200_show_uint(72, 248, ring_second_turn_frames, 2);
        ips200_show_string(104, 248, "BF:");
        ips200_show_uint(128, 248, boundary_assist_frames, 2);
        ips200_show_string(160, 248, "BC:");
        ips200_show_uint(184, 248, both_edges_lost, 1);
        ips200_show_string(200, 248, "BR:");
        ips200_show_uint(224, 248, both_edges_recovered, 1);

        /* Speed, PID command, ring command and virtual-boundary flag. */
        ips200_show_string(0, 264, "V:");
        ips200_show_uint(16, 264, image_base_speed_percent, 2);
        ips200_show_string(48, 264, "O:");
        ips200_show_uint(64, 264, (uint32)steering_command, 2);
        ips200_show_string(96, 264, "RC:");
        ips200_show_uint(120, 264, (uint32)ring_turn_command, 2);
        ips200_show_string(152, 264, "BA:");
        ips200_show_uint(176, 264, boundary_assist_active, 1);

        /* Ring wait continuity, adaptive timeout and force ratio. */
        ips200_show_string(0, 280, "MC:");
        ips200_show_uint(24, 280, reference_near_center, 1);
        ips200_show_string(48, 280, "ML:");
        ips200_show_uint(72, 280, reference_lost_frames, 2);
        ips200_show_string(104, 280, "TO:");
        ips200_show_uint(128, 280, current_second_gap_timeout_frames, 3);
        ips200_show_string(176, 280, "RP:");
        ips200_show_uint(200, 280, RING_FORCE_RATIO_PERCENT, 3);

        ips200_show_string(0, 296, "LE:");
        ips200_show_uint(24, 296, current_left_edge_average, 3);
        ips200_show_string(64, 296, "RE:");
        ips200_show_uint(88, 296, current_right_edge_average, 3);
        ips200_show_string(128, 296, "PZ:");
        ips200_show_uint(152, 296, control_z, 3);
        ips200_show_string(192, 296, "ZX:");
        ips200_show_uint(216, 296, zhuanxiang, 3);
#endif
    }
}

#pragma section all restore
