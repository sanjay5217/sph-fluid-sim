#import "GameViewController.h"
#import "Renderer.h"
#import "ShaderTypes.h"

typedef NS_ENUM(NSInteger, SettingTag) {
    SettingTagParticleCount = 1,
    SettingTagPressureStiffness,
    SettingTagViscosity,
};

@implementation GameViewController
{
    MTKView *_view;

    Renderer *_renderer;

    NSMutableDictionary<NSNumber *, NSTextField *> *_valueLabels;
    NSTextField *_derivedLabel;
}

- (void)viewDidLoad
{
    [super viewDidLoad];

    _view = (MTKView *)self.view;

    _view.device = MTLCreateSystemDefaultDevice();

    if(!_view.device)
    {
        NSLog(@"Metal is not supported on this device");
        self.view = [[NSView alloc] initWithFrame:self.view.frame];
        return;
    }

    _renderer = [[Renderer alloc] initWithMetalKitView:_view];

    [_renderer mtkView:_view drawableSizeWillChange:_view.drawableSize];

    _view.delegate = _renderer;

    [self buildSettingsPanel];
}

#pragma mark - Settings panel

- (void)buildSettingsPanel
{
    _valueLabels = [NSMutableDictionary dictionary];

    NSVisualEffectView *panel = [[NSVisualEffectView alloc] initWithFrame:NSZeroRect];
    panel.material = NSVisualEffectMaterialHUDWindow;
    panel.blendingMode = NSVisualEffectBlendingModeWithinWindow;
    panel.state = NSVisualEffectStateActive;
    panel.wantsLayer = YES;
    panel.layer.cornerRadius = 8.0;
    panel.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:panel];

    NSStackView *stack = [[NSStackView alloc] initWithFrame:NSZeroRect];
    stack.orientation = NSUserInterfaceLayoutOrientationVertical;
    stack.alignment = NSLayoutAttributeLeading;
    stack.spacing = 6.0;
    stack.translatesAutoresizingMaskIntoConstraints = NO;
    [panel addSubview:stack];

    [stack addArrangedSubview:[self rowWithTitle:@"Particles"
                                             tag:SettingTagParticleCount
                                        minValue:500 maxValue:10000
                                           value:_renderer.particleCount]];
    [stack addArrangedSubview:[self rowWithTitle:@"Stiffness k"
                                             tag:SettingTagPressureStiffness
                                        minValue:25 maxValue:800
                                           value:_renderer.pressureStiffness]];
    [stack addArrangedSubview:[self rowWithTitle:@"Viscosity"
                                             tag:SettingTagViscosity
                                        minValue:0 maxValue:80
                                           value:_renderer.viscosity]];

    _derivedLabel = [NSTextField labelWithString:@""];
    _derivedLabel.font = [NSFont monospacedSystemFontOfSize:10.0 weight:NSFontWeightRegular];
    _derivedLabel.textColor = [NSColor secondaryLabelColor];
    _derivedLabel.translatesAutoresizingMaskIntoConstraints = NO;
    [stack addArrangedSubview:_derivedLabel];

    NSButton *resetButton = [NSButton buttonWithTitle:@"Reset fluid"
                                               target:self
                                               action:@selector(resetTapped:)];
    resetButton.controlSize = NSControlSizeSmall;
    resetButton.translatesAutoresizingMaskIntoConstraints = NO;
    [stack addArrangedSubview:resetButton];

    [NSLayoutConstraint activateConstraints:@[
        [panel.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:12.0],
        [panel.topAnchor constraintEqualToAnchor:self.view.topAnchor constant:12.0],
        [panel.widthAnchor constraintEqualToConstant:288.0],

        [stack.leadingAnchor constraintEqualToAnchor:panel.leadingAnchor constant:12.0],
        [stack.trailingAnchor constraintEqualToAnchor:panel.trailingAnchor constant:-12.0],
        [stack.topAnchor constraintEqualToAnchor:panel.topAnchor constant:12.0],
        [stack.bottomAnchor constraintEqualToAnchor:panel.bottomAnchor constant:-12.0],
    ]];

    [self refreshLabels];
}

- (NSView *)rowWithTitle:(NSString *)title
                     tag:(SettingTag)tag
                minValue:(double)minValue
                maxValue:(double)maxValue
                   value:(double)value
{
    NSTextField *titleLabel = [NSTextField labelWithString:title];
    titleLabel.font = [NSFont systemFontOfSize:11.0];
    titleLabel.textColor = [NSColor labelColor];

    NSTextField *valueLabel = [NSTextField labelWithString:@""];
    valueLabel.font = [NSFont monospacedSystemFontOfSize:11.0 weight:NSFontWeightMedium];
    valueLabel.textColor = [NSColor labelColor];
    valueLabel.alignment = NSTextAlignmentRight;
    [valueLabel setContentHuggingPriority:NSLayoutPriorityDefaultHigh
                           forOrientation:NSLayoutConstraintOrientationHorizontal];
    _valueLabels[@(tag)] = valueLabel;

    NSSlider *slider = [NSSlider sliderWithValue:value
                                        minValue:minValue
                                        maxValue:maxValue
                                          target:self
                                          action:@selector(sliderChanged:)];
    slider.tag = tag;
    slider.controlSize = NSControlSizeSmall;
    slider.continuous = YES;

    NSStackView *header = [NSStackView stackViewWithViews:@[titleLabel, valueLabel]];
    header.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    header.distribution = NSStackViewDistributionFill;

    NSStackView *row = [NSStackView stackViewWithViews:@[header, slider]];
    row.orientation = NSUserInterfaceLayoutOrientationVertical;
    row.alignment = NSLayoutAttributeLeading;
    row.spacing = 1.0;

    [header.widthAnchor constraintEqualToAnchor:row.widthAnchor].active = YES;
    [slider.widthAnchor constraintEqualToAnchor:row.widthAnchor].active = YES;

    return row;
}

- (void)sliderChanged:(NSSlider *)sender
{
    switch ((SettingTag)sender.tag) {
        case SettingTagParticleCount:
            _renderer.particleCount = (NSUInteger)lround(sender.doubleValue);
            break;
        case SettingTagPressureStiffness:
            _renderer.pressureStiffness = (float)sender.doubleValue;
            break;
        case SettingTagViscosity:
            _renderer.viscosity = (float)sender.doubleValue;
            break;
    }
    [self refreshLabels];
}

- (void)resetTapped:(id)sender
{
    [_renderer resetSimulation];
}

- (void)refreshLabels
{
    _valueLabels[@(SettingTagParticleCount)].stringValue =
        [NSString stringWithFormat:@"%lu", (unsigned long)_renderer.particleCount];
    _valueLabels[@(SettingTagPressureStiffness)].stringValue =
        [NSString stringWithFormat:@"%.0f", _renderer.pressureStiffness];
    _valueLabels[@(SettingTagViscosity)].stringValue =
        [NSString stringWithFormat:@"%.1f", _renderer.viscosity];

    float neighbours = _renderer.neighbourCount;
    float dt = _renderer.timestep;
    float limit = _renderer.stableTimestepLimit;

    NSString *neighbourNote = (neighbours < 15.0f) ? @" too few"
                            : (neighbours > 50.0f) ? @" too many"
                                                   : @" ok";

    _derivedLabel.stringValue = [NSString stringWithFormat:
        @"smoothing h   %.3f m base, adaptive\n"
         "rest density  %.0f (fixed)\n"
         "rest spacing  %.4f m\n"
         "neighbours    %.0f (want 20-40)%@\n"
         "fluid covers  %.2f m2 of %.0f (%.0f%% full)\n"
         "substeps      %lu -> %.2f ms (%.0f%% of limit)",
        _renderer.smoothingRadius,
        _renderer.restDensity,
        _renderer.restSpacing,
        neighbours, neighbourNote,
        _renderer.occupiedArea,
        (double)(kContainerWidth * kContainerHeight),
        100.0f * _renderer.fillFraction,
        (unsigned long)_renderer.substepsPerFrame,
        dt * 1000.0f,
        limit > 0.0f ? 100.0f * dt / limit : 0.0f];
}

@end
