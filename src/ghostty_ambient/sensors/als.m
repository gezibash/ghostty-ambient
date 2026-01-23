// als.m - Ambient Light Sensor CLI for macOS
// Supports both built-in MacBook sensors and external displays (e.g., Studio Display)
// Compile: clang -framework IOKit -framework Foundation -framework CoreFoundation -F /System/Library/PrivateFrameworks -framework BezelServices -o als als.m

#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <IOKit/hid/IOHIDManager.h>
#import <IOKit/hid/IOHIDDevice.h>
#import <IOKit/hid/IOHIDElement.h>
#import <IOKit/hid/IOHIDValue.h>
#import <math.h>

typedef struct __IOHIDEvent *IOHIDEventRef;
typedef void *IOHIDServiceClientRef;

#define kAmbientLightSensorEvent 12
#define IOHIDEventFieldBase(type) (type << 16)

// HID Sensor Usage Page and Usage IDs
#define kHIDPage_Sensor 0x20
#define kHIDUsage_Sensor_AmbientLight 0x41
#define kHIDUsage_Sensor_Data_LightIlluminance 0x04D1  // The actual illuminance value element

extern IOHIDEventRef IOHIDServiceClientCopyEvent(IOHIDServiceClientRef, int64_t, int32_t, int64_t);
extern double IOHIDEventGetFloatValue(IOHIDEventRef, int32_t);
extern IOHIDServiceClientRef ALCALSCopyALSServiceClient(void);

// Check if a device is an ambient light sensor
static BOOL isAmbientLightSensor(IOHIDDeviceRef device) {
    // Check primary usage
    int32_t usagePage = [(NSNumber *)IOHIDDeviceGetProperty(device, CFSTR(kIOHIDPrimaryUsagePageKey)) intValue];
    int32_t usage = [(NSNumber *)IOHIDDeviceGetProperty(device, CFSTR(kIOHIDPrimaryUsageKey)) intValue];

    // Usage page 32 (0x20) is Sensors, usage 65 (0x41) is AmbientLight
    // Also check for usage 138 (0x8A) which some displays use
    if (usagePage == kHIDPage_Sensor && (usage == kHIDUsage_Sensor_AmbientLight || usage == 0x8A)) {
        return YES;
    }
    return NO;
}

// Check if this is an external display (not built-in)
static BOOL isExternalDisplay(IOHIDDeviceRef device) {
    NSString *transport = (__bridge NSString *)IOHIDDeviceGetProperty(device, CFSTR(kIOHIDTransportKey));
    NSString *product = (__bridge NSString *)IOHIDDeviceGetProperty(device, CFSTR(kIOHIDProductKey));

    // USB-connected displays are external
    if ([transport isEqualToString:@"USB"]) {
        return YES;
    }

    // Check for known external display names
    if (product && ([product containsString:@"Studio Display"] ||
                    [product containsString:@"Pro Display"] ||
                    [product containsString:@"LG UltraFine"])) {
        return YES;
    }

    return NO;
}

// Convert HID unit exponent (4-bit signed nibble) to actual exponent
// Values 0-7 are positive (0 to 7), values 8-15 are negative (-8 to -1)
static int convertUnitExponent(uint32_t unitExp) {
    if (unitExp <= 7) {
        return (int)unitExp;
    } else {
        return (int)unitExp - 16;  // 8->-8, 9->-7, ..., 15->-1
    }
}

// Read illuminance from a HID ambient light sensor device
static double readHIDSensorValue(IOHIDDeviceRef device) {
    CFArrayRef elements = IOHIDDeviceCopyMatchingElements(device, NULL, kIOHIDOptionsTypeNone);
    if (!elements) return -1;

    double lux = -1;
    CFIndex count = CFArrayGetCount(elements);

    for (CFIndex i = 0; i < count; i++) {
        IOHIDElementRef element = (IOHIDElementRef)CFArrayGetValueAtIndex(elements, i);
        uint32_t usagePage = IOHIDElementGetUsagePage(element);
        uint32_t usage = IOHIDElementGetUsage(element);

        // Look for illuminance data field (usage page 0x20, usage 0x04D1)
        // This is the standard HID sensor usage for light illuminance in lux
        if (usagePage == kHIDPage_Sensor && usage == kHIDUsage_Sensor_Data_LightIlluminance) {
            IOHIDValueRef value = NULL;
            if (IOHIDDeviceGetValue(device, element, &value) == kIOReturnSuccess && value) {
                CFIndex rawValue = IOHIDValueGetIntegerValue(value);

                // Get the unit exponent to properly scale the value
                // Unit exponent is a 4-bit signed nibble that represents power of 10
                uint32_t unitExp = IOHIDElementGetUnitExponent(element);
                int exponent = convertUnitExponent(unitExp);

                // Apply the unit exponent: value * 10^exponent
                lux = (double)rawValue * pow(10.0, exponent);
                break;
            }
        }
    }

    CFRelease(elements);
    return lux;
}

// Try to read from external display ambient light sensors (like Studio Display)
static double readExternalDisplaySensor(void) {
    IOHIDManagerRef manager = IOHIDManagerCreate(kCFAllocatorDefault, kIOHIDOptionsTypeNone);
    if (!manager) return -1;

    // Match specifically ambient light sensor devices (usage page 0x20, usage 0x41)
    // This ensures we get devices that have the illuminance data element
    NSDictionary *matching = @{
        @kIOHIDDeviceUsagePageKey: @(kHIDPage_Sensor),
        @kIOHIDPrimaryUsageKey: @(kHIDUsage_Sensor_AmbientLight)
    };

    IOHIDManagerSetDeviceMatching(manager, (__bridge CFDictionaryRef)matching);
    IOHIDManagerOpen(manager, kIOHIDOptionsTypeNone);

    CFSetRef deviceSet = IOHIDManagerCopyDevices(manager);
    double lux = -1;

    if (deviceSet) {
        CFIndex count = CFSetGetCount(deviceSet);
        if (count > 0) {
            IOHIDDeviceRef *devices = malloc(sizeof(IOHIDDeviceRef) * count);
            CFSetGetValues(deviceSet, (const void **)devices);

            // Prefer external display sensors over built-in
            IOHIDDeviceRef externalDevice = NULL;
            IOHIDDeviceRef anyDevice = NULL;

            for (CFIndex i = 0; i < count; i++) {
                IOHIDDeviceRef device = devices[i];
                anyDevice = device;
                if (isExternalDisplay(device)) {
                    externalDevice = device;
                    break;
                }
            }

            // Try external first, then any ALS
            IOHIDDeviceRef targetDevice = externalDevice ? externalDevice : anyDevice;
            if (targetDevice) {
                // Try to open and read from the device
                if (IOHIDDeviceOpen(targetDevice, kIOHIDOptionsTypeNone) == kIOReturnSuccess) {
                    lux = readHIDSensorValue(targetDevice);
                    IOHIDDeviceClose(targetDevice, kIOHIDOptionsTypeNone);
                }
            }

            free(devices);
        }
        CFRelease(deviceSet);
    }

    IOHIDManagerClose(manager, kIOHIDOptionsTypeNone);
    CFRelease(manager);

    return lux;
}

// Read from built-in sensor using BezelServices (modern Macs)
static double readModernSensor(void) {
    IOHIDServiceClientRef client = ALCALSCopyALSServiceClient();
    if (!client) return -1;

    IOHIDEventRef event = IOHIDServiceClientCopyEvent(client, kAmbientLightSensorEvent, 0, 0);
    if (!event) {
        CFRelease(client);
        return -1;
    }

    double lux = IOHIDEventGetFloatValue(event, IOHIDEventFieldBase(kAmbientLightSensorEvent));
    CFRelease(event);
    CFRelease(client);
    return lux;
}

// Read from built-in sensor using legacy IOKit method (older Macs)
static double readLegacySensor(void) {
    io_service_t service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleLMUController"));
    if (!service) return -1;

    io_connect_t port;
    if (IOServiceOpen(service, mach_task_self(), 0, &port) != KERN_SUCCESS) {
        IOObjectRelease(service);
        return -1;
    }
    IOObjectRelease(service);

    uint32_t outputs = 2;
    uint64_t values[2];

    if (IOConnectCallMethod(port, 0, nil, 0, nil, 0, values, &outputs, nil, 0) == KERN_SUCCESS) {
        double v = (double)(3 * values[0] / 100000 - 1.5);
        IOServiceClose(port);
        return (v > 0) ? v : 0.0;
    }

    IOServiceClose(port);
    return -1;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        double lux = -1;

        // Strategy: Try external display sensors first (like Studio Display),
        // then fall back to built-in sensors

        // 1. Try HID sensor devices (catches external displays)
        lux = readExternalDisplaySensor();

        // 2. If no external sensor or low/zero value, try built-in modern sensor
        if (lux <= 0) {
            double builtInLux = readModernSensor();
            // Use built-in if it returns a valid value
            if (builtInLux > 0) {
                lux = builtInLux;
            } else if (lux < 0) {
                lux = builtInLux;  // Use whatever we got if external failed
            }
        }

        // 3. Try legacy sensor as last resort
        if (lux < 0) {
            lux = readLegacySensor();
        }

        if (lux < 0) {
            fprintf(stderr, "error: could not read ambient light sensor\n");
            return 1;
        }

        printf("%.0f\n", lux);
        return 0;
    }
}
